"""Train the stroke outline model and report held-out glyph IoU.

    python -m cnn_chinese_hw.stroke_model.train [--epochs 200] [--device cuda]

The baseline is a uniform-width stroke along Tomoe's own points whose glyph box
is fitted to the true outline's box: today's renderer, generously given the
true proportions it cannot know. The model gets no such help.
"""
import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch

from . import dataset as D
from . import representation as R
from .infer import CHECKPOINT
from .model import StrokeOutlineModel
BASELINE_HALF_WIDTH = 30 / 1024


def batch(glyphs, device, jitter=0.0):
    s = max(len(g['inputs']) for g in glyphs)
    shape = (len(glyphs), s, R.N)
    inputs = np.zeros(shape + (2,), np.float32)
    centre = np.zeros(shape + (2,), np.float32)
    left, right = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    mask = np.zeros(shape[:2], bool)
    for i, g in enumerate(glyphs):
        k = len(g['inputs'])
        inputs[i, :k], centre[i, :k], left[i, :k], right[i, :k] = g['inputs'], g['centre'], g['left'], g['right']
        mask[i, :k] = True
    if jitter:
        inputs += np.random.normal(0, jitter, inputs.shape).astype(np.float32) * mask[..., None, None]
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(inputs), t(mask), t(centre), t(left), t(right)


def loss_fn(pred, mask, centre, left, right):
    pc, pl, pr = pred
    m = mask[..., None].float()
    n = m.sum() * R.N
    l_centre = ((pc - centre).abs().sum(-1) * m).sum() / n
    l_width = (((pl - left).abs() + (pr - right).abs()) * m).sum() / n
    return (l_centre + l_width) * 1024, l_centre * 1024, l_width * 1024


def glyph_polygons(centre, left, right):
    return [R.decode_polygon(c * 1024, l * 1024, r * 1024) for c, l, r in zip(centre, left, right)]


def baseline_polygons(g):
    """Uniform-width Tomoe strokes, glyph box fitted to the true outline's."""
    pts = np.concatenate(g['inputs'])
    lo, hi = pts.min(0), pts.max(0)
    gt = np.concatenate(g['centre'])
    glo, ghi = gt.min(0), gt.max(0)
    scale = (ghi - glo) / np.where(hi - lo < 1e-6, 1, hi - lo)
    out = []
    for s in g['inputs']:
        c = (s - lo) * scale + glo
        w = np.full(R.N, BASELINE_HALF_WIDTH)
        out.append(R.decode_polygon(c * 1024, w * 1024, w * 1024))
    return out


@torch.no_grad()
def evaluate(model, glyphs, device):
    model.eval()
    ious, base, losses = [], [], []
    for i in range(0, len(glyphs), 64):
        chunk = glyphs[i:i + 64]
        inputs, mask, centre, left, right = batch(chunk, device)
        pred = model(inputs, mask)
        losses.append(loss_fn(pred, mask, centre, left, right)[0].item() * len(chunk))
        pc, pl, pr = (p.cpu().numpy() for p in pred)
        for j, g in enumerate(chunk):
            k = len(g['inputs'])
            truth = glyph_polygons(g['centre'], g['left'], g['right'])
            ious.append(R.glyph_iou(truth, glyph_polygons(pc[j, :k], pl[j, :k], pr[j, :k])))
            base.append(R.glyph_iou(truth, baseline_polygons(g)))
    model.train()
    return {'loss_em': sum(losses) / len(glyphs), 'iou': float(np.mean(ious)), 'baseline_iou': float(np.mean(base)),
            'iou_p10': float(np.percentile(ious, 10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=CHECKPOINT)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    glyphs = D.load()
    train = [g for g in glyphs if g['split'] == 'train']
    val = [g for g in glyphs if g['split'] == 'val']
    test = [g for g in glyphs if g['split'] == 'test']
    print(f'pairs: train {len(train)}, val {len(val)}, test {len(test)}', flush=True)

    model = StrokeOutlineModel().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    best, history, t0 = None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        # Similar stroke counts per batch keep padding low.
        order = sorted(range(len(train)), key=lambda i: len(train[i]['inputs']) + random.random() * 4)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        random.shuffle(batches)
        total = 0.0
        for idx in batches:
            inputs, mask, centre, left, right = batch([train[i] for i in idx], args.device, jitter=0.004)
            loss = loss_fn(model(inputs, mask), mask, centre, left, right)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(idx)
        if epoch % 20 == 0 or epoch == args.epochs:
            metrics = evaluate(model, val, args.device)
            history.append({'epoch': epoch, 'train_loss_em': total / len(train), **metrics})
            print(json.dumps(history[-1]), f'{time.time() - t0:.0f}s', flush=True)
            if best is None or metrics['iou'] > best[0]:
                best = (metrics['iou'], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    test_metrics = evaluate(model, test, args.device)
    print('best epoch', best[2], 'test', json.dumps(test_metrics), flush=True)
    data_hash = hashlib.sha256(open(D.CACHE, 'rb').read()).hexdigest()
    torch.save({'state_dict': best[1], 'epoch': best[2], 'val': history, 'test': test_metrics,
                'pairs_sha256': data_hash, 'args': vars(args), 'n': R.N}, args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
