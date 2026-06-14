"""Inference-time handwriting recognizer (PyTorch).

Loads one or more checkpoints produced by ``train.py`` and turns a list of
online strokes into ranked character candidates. This is the PyTorch
replacement for the old ``TFLiteRecognizer`` and keeps the same public method,
``get_candidates_list(strokes_list, n_cands) -> [(score, ordinal), ...]``.

Ensemble of license-separated models
-------------------------------------
To avoid co-mingling incompatibly-licensed training corpora in one set of
weights (see ``config.LICENSE_GROUPS`` and ``THIRD_PARTY_DATA.md``), the project
trains a separate checkpoint per license group. At inference those models are
combined here by **summing their temperature-calibrated probabilities** over the
union of their class sets, then renormalising. Because each model's softmax sums
to 1 and we divide by the model count, the combined scores still form a proper
distribution that sums to 1. Each weight file stays separately licensed; only
their numeric outputs meet, at runtime.

When constructed with an explicit ``checkpoint_path`` a single model is used
(the original behaviour). With no path, all available per-group checkpoints
(``hw_model.<group>.pt``) are discovered and ensembled, falling back to the
legacy single ``hw_model.pt`` if no split checkpoints exist.
"""

import os
import torch
import threading
import torch.nn.functional as F

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.recognizer.model import build_model
from cnn_chinese_hw.recognizer.config import DataConfig, ModelConfig
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter


_MISSING_MODEL_MSG = (
    "No trained checkpoint found at {path}.\n"
    "Train one first with:\n"
    "    python -m cnn_chinese_hw.recognizer.train --license-group permissive\n"
    "    python -m cnn_chinese_hw.recognizer.train --license-group ccbysa\n"
    "(or `--smoke` for a quick pipeline check)."
)


def discover_checkpoints():
    """Existing per-group checkpoints, else the legacy single checkpoint."""
    paths = [config.checkpoint_path_for(g) for g in config.LICENSE_GROUPS]
    paths = [p for p in paths if os.path.exists(p)]
    if paths:
        return paths
    if os.path.exists(config.CHECKPOINT_PATH):
        return [config.CHECKPOINT_PATH]
    return []


class _LoadedModel:
    """One checkpoint: its network, class list and calibration temperature."""

    def __init__(self, path, device, use_ema=True):
        self.path = path
        self.device = torch.device(device)
        # The checkpoint contains only tensors + plain containers, so it loads
        # under the safe (weights_only) unpickler.
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        self.classes = list(ckpt['classes'])
        self.data_cfg = DataConfig(**ckpt['data_cfg'])
        model_cfg = ModelConfig(**ckpt['model_cfg'])

        self.model = build_model(model_cfg).to(self.device)
        state = ckpt.get('ema_state') if use_ema else None
        self.model.load_state_dict(state or ckpt['model_state'])
        self.model.eval()

        # Post-hoc calibration temperature (Guo et al., ICML 2017); 1.0 = none.
        self.temperature = float(ckpt.get('temperature', 1.0)) or 1.0

    @torch.no_grad()
    def probs(self, x):
        """Mean calibrated probability over the (possibly TTA-augmented) batch,
        one value per class in ``self.classes``."""
        logits = self.model(x.to(self.device))
        return F.softmax(logits / self.temperature, dim=1).mean(dim=0).cpu()


class HandwritingRecognizer:
    def __init__(self, checkpoint_path=None, device='cpu', use_ema=True):
        if checkpoint_path is not None:
            paths = [checkpoint_path]
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    _MISSING_MODEL_MSG.format(path=checkpoint_path))
        else:
            paths = discover_checkpoints()
            if not paths:
                raise FileNotFoundError(
                    _MISSING_MODEL_MSG.format(path=config.CHECKPOINT_PATH))

        self.device = torch.device(device)
        self.models = [_LoadedModel(p, device, use_ema) for p in paths]

        # Global vocabulary = union of every model's classes, and, per model, a
        # tensor mapping its local class index -> global vocab index. Lets us
        # accumulate each model's probabilities with a single index_add_.
        ord_to_global = {}
        self.vocab = []
        for m in self.models:
            for o in m.classes:
                if o not in ord_to_global:
                    ord_to_global[o] = len(self.vocab)
                    self.vocab.append(o)
        for m in self.models:
            m.global_idx = torch.tensor([ord_to_global[o] for o in m.classes],
                                        dtype=torch.long)

        # Back-compat attribute (single-model callers read it); for an ensemble
        # it is the first model's temperature.
        self.temperature = self.models[0].temperature
        # Inference is single-instance and may be called from request threads.
        self.lock = threading.Lock()

    @property
    def classes(self):
        return self.vocab

    def _encode(self, aug, image_size, do_augment):
        rastered = aug.raster_strokes(image_size=image_size, do_augment=do_augment)
        x = torch.from_numpy(rastered).permute(2, 0, 1).float().div_(255.0)
        return x.unsqueeze(0)

    @torch.no_grad()
    def get_candidates_list(self, strokes_list, n_cands=20, tta=0):
        """Rank character candidates for a set of online strokes.

        :param strokes_list: ``[[(x, y), ...], ...]`` (one inner list per stroke)
        :param n_cands: number of candidates to return
        :param tta: number of extra randomly-augmented renders to average over
                    (test-time augmentation). 0 = a single clean render (fast,
                    the default for low-latency serving). Averaging the softmax
                    over augmented inputs is a long-standing accuracy trick
                    (Krizhevsky et al., NeurIPS 2012; Simonyan & Zisserman,
                    ICLR 2015) and mirrors what the original demo GUI did.
        :return: ``[(score, ordinal), ...]`` sorted by descending score.
                 Use ``chr(ordinal)`` to get the character. With multiple models
                 the score is the mean of their calibrated probabilities.
        """
        # All models share the same input geometry (config.IMAGE_SIZE), so the
        # raster is built once and fed to each model.
        image_size = self.models[0].data_cfg.image_size
        aug = HWStrokesAugmenter(strokes_list, find_vertices=True)
        batch = [self._encode(aug, image_size, do_augment=False)]
        batch += [self._encode(aug, image_size, do_augment=True)
                  for _ in range(max(tta, 0))]
        x = torch.cat(batch, dim=0)

        acc = torch.zeros(len(self.vocab))
        with self.lock:
            for m in self.models:
                acc.index_add_(0, m.global_idx, m.probs(x))
        # Divide by model count: each model's mass is 1, so the union still sums
        # to 1 and single-model behaviour is unchanged.
        acc /= len(self.models)

        n = min(n_cands, acc.numel())
        scores, idx = torch.topk(acc, n)
        return [(float(s), self.vocab[int(i)]) for s, i in zip(scores, idx)]


if __name__ == '__main__':
    rec = HandwritingRecognizer()
    print(f"Loaded {len(rec.models)} model(s); {len(rec.vocab)} classes total.")
    # 我 (ord 25105)
    print(rec.get_candidates_list(
        [[(208, 0), (199, 119), (94, 341)],
         [(0, 461), (781, 520), (915, 520), (999, 479)],
         [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)],
         [(303, 514), (94, 766)],
         [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)],
         [(716, 628), (462, 916)],
         [(696, 101), (771, 155), (835, 251)]]
    ))
