"""Train the DirectMap SE-ResNet handwriting recognizer (PyTorch).

Usage
-----
Full training run (produces data/hw_model.pt)::

    python -m cnn_chinese_hw.recognizer.train

Fast smoke test on a tiny subset (verifies the whole pipeline end to end)::

    python -m cnn_chinese_hw.recognizer.train --smoke

Common overrides::

    python -m cnn_chinese_hw.recognizer.train --epochs 200 --batch-size 384

Training recipe (see docs/ARCHITECTURE.md for citations):
  * AdamW with decoupled weight decay (Loshchilov & Hutter, ICLR 2019)
  * cosine learning-rate schedule with linear warm-up (Loshchilov & Hutter,
    SGDR, ICLR 2017)
  * label smoothing (Szegedy et al., CVPR 2016; Müller et al., NeurIPS 2019)
  * automatic mixed precision (torch.amp)
  * exponential moving average of the weights for the evaluated model
  * early stopping on validation accuracy
"""

import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.recognizer.dataset import build_datasets
from cnn_chinese_hw.recognizer.model import build_model


def seed_everything(seed: int):
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id):
    # Give every DataLoader worker a distinct but deterministic seed, so the
    # on-the-fly augmentation is varied across workers yet reproducible.
    base = torch.initial_seed() % 2 ** 31
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % 2 ** 31)


def mixup_cutmix(x, y, train_cfg):
    """Apply Mixup (Zhang et al., ICLR 2018) or CutMix (Yun et al., ICCV 2019)
    to a batch with probability ``mix_prob``. Returns ``(x, y_a, y_b, lam)``;
    the mixed loss is ``lam*loss(y_a) + (1-lam)*loss(y_b)``. With no mixing,
    ``y_a == y_b`` and ``lam == 1``."""
    if ((train_cfg.mixup_alpha <= 0 and train_cfg.cutmix_alpha <= 0)
            or random.random() >= train_cfg.mix_prob):
        return x, y, y, 1.0

    perm = torch.randperm(x.size(0), device=x.device)
    use_cutmix = train_cfg.cutmix_alpha > 0 and (
        train_cfg.mixup_alpha <= 0 or random.random() < 0.5)

    if use_cutmix:
        lam = float(np.random.beta(train_cfg.cutmix_alpha, train_cfg.cutmix_alpha))
        h, w = x.shape[2], x.shape[3]
        r = math.sqrt(1.0 - lam)
        cw, ch = int(w * r), int(h * r)
        cx, cy = int(np.random.randint(w)), int(np.random.randint(h))
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))  # true mixed-area ratio
    else:
        lam = float(np.random.beta(train_cfg.mixup_alpha, train_cfg.mixup_alpha))
        x = lam * x + (1.0 - lam) * x[perm]

    return x, y, y[perm], lam


def make_balanced_sampler(train_ds, class_counts, beta):
    """WeightedRandomSampler that re-weights by the inverse *effective number*
    of samples per class (Cui et al., CVPR 2019): ``(1 - beta) / (1 - beta**n)``.
    Counters the imbalance from characters that appear in multiple scripts or
    have several hand-drawn copies."""
    eff_num = [(1.0 - beta ** max(n, 1)) / (1.0 - beta) for n in class_counts]
    class_w = [(1.0 / e) if e > 0 else 0.0 for e in eff_num]
    weights = [class_w[label] for label in train_ds.labels()]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ModelEMA:
    """Exponential moving average of model weights (the evaluated/exported copy).

    The decay is optionally warmed up -- ``min(decay, (1 + step)/(10 + step))``
    -- so the averaged model tracks the (rapidly improving) net closely at the
    start and only smooths heavily once training stabilises. This avoids the
    averaged weights lagging near random initialisation for the first epochs
    (Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017,
    arXiv:1703.01780)."""

    def __init__(self, model, decay=0.999, warmup=True):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = self.decay
        if self.warmup:
            d = min(d, (1 + self.step) / (10 + self.step))
        for ema_p, p in zip(self.ema.state_dict().values(),
                            model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                ema_p.copy_(p)


def make_scheduler(optimizer, train_cfg, steps_per_epoch):
    warmup_steps = train_cfg.warmup_epochs * steps_per_epoch
    total_steps = train_cfg.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, topk=10):
    """Returns ``(top1, topk)`` accuracy."""
    model.eval()
    top1 = topk_correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        k = min(topk, logits.shape[1])
        _, pred_k = logits.topk(k, dim=1)
        correct = pred_k == y[:, None]
        top1 += correct[:, 0].sum().item()
        topk_correct += correct.any(dim=1).sum().item()
        total += y.numel()
    total = max(total, 1)
    return top1 / total, topk_correct / total


def save_checkpoint(path, model, ema, store, model_cfg, data_cfg,
                    val_top1, val_topk, epoch):
    torch.save({
        'model_state': model.state_dict(),
        'ema_state': ema.ema.state_dict() if ema is not None else None,
        'classes': store.classes,
        'model_cfg': config.config_to_dict(model_cfg),
        'data_cfg': config.config_to_dict(data_cfg),
        'val_acc': val_top1,
        'val_topk': val_topk,
        'epoch': epoch,
        'format': 1,
    }, path)


def train(data_cfg, model_cfg, train_cfg, cache=True):
    seed_everything(train_cfg.seed)
    generator = torch.Generator().manual_seed(train_cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    store, train_ds, val_ds = build_datasets(data_cfg, cache=cache)
    model_cfg.num_classes = store.num_classes

    if train_cfg.balanced_sampling:
        sampler = make_balanced_sampler(train_ds, store.class_counts, train_cfg.cb_beta)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=data_cfg.num_workers,
        pin_memory=(device == 'cuda'), drop_last=True,
        persistent_workers=data_cfg.num_workers > 0,
        worker_init_fn=_worker_init_fn, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=(device == 'cuda'),
    )

    model = build_model(model_cfg).to(device)
    print(f"Model: {model.num_parameters() / 1e6:.2f}M params, "
          f"{model_cfg.num_classes} classes")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = make_scheduler(optimizer, train_cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.amp and device == 'cuda')
    ema = ModelEMA(model, train_cfg.ema_decay, warmup=train_cfg.ema_warmup)

    best_topk = 0.0
    epochs_no_improve = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x, y_a, y_b, lam = mixup_cutmix(x, y, train_cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running += loss.item()

        val_top1, val_topk = evaluate(ema.ema, val_loader, device, train_cfg.topk)
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1:3d}/{train_cfg.epochs} "
              f"loss={running / len(train_loader):.4f} "
              f"val_top1={val_top1:.4f} val_top{train_cfg.topk}={val_topk:.4f} "
              f"lr={lr_now:.2e} ({time.time() - t0:.1f}s)")

        # Select on top-k: an IME is judged by whether the character is in the
        # candidate list, not only when it is ranked first.
        if val_topk > best_topk:
            best_topk = val_topk
            epochs_no_improve = 0
            save_checkpoint(config.CHECKPOINT_PATH, model, ema, store,
                            model_cfg, data_cfg, val_top1, val_topk, epoch)
            print(f"  -> saved checkpoint (val_top{train_cfg.topk}={val_topk:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best val_top{train_cfg.topk}={best_topk:.4f})")
                break

    print(f"Done. Best val_top{train_cfg.topk}={best_topk:.4f}. "
          f"Checkpoint: {config.CHECKPOINT_PATH}")
    return best_topk


def main():
    data_cfg, model_cfg, train_cfg = config.default_configs()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=train_cfg.epochs)
    p.add_argument('--batch-size', type=int, default=train_cfg.batch_size)
    p.add_argument('--lr', type=float, default=train_cfg.lr)
    p.add_argument('--workers', type=int, default=data_cfg.num_workers)
    p.add_argument('--no-cache', action='store_true',
                   help="don't read/write the parsed-stroke cache")
    p.add_argument('--smoke', action='store_true',
                   help="tiny subset + few epochs to verify the pipeline")
    args = p.parse_args()

    train_cfg.epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.lr = args.lr
    data_cfg.num_workers = args.workers

    if args.smoke:
        data_cfg.small_sample_only = True
        data_cfg.small_sample_size = 60
        data_cfg.augmentations_per_sample = 4
        train_cfg.epochs = 3
        train_cfg.warmup_epochs = 1
        train_cfg.early_stop_patience = 99

    train(data_cfg, model_cfg, train_cfg, cache=not args.no_cache)


if __name__ == '__main__':
    main()
