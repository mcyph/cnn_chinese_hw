"""Inference-time handwriting recognizer (PyTorch).

Loads a checkpoint produced by ``train.py`` and turns a list of online strokes
into ranked character candidates. This is the PyTorch replacement for the old
``TFLiteRecognizer`` and keeps the same public method,
``get_candidates_list(strokes_list, n_cands) -> [(score, ordinal), ...]``.
"""

import os
import threading

import torch
import torch.nn.functional as F

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.recognizer.config import DataConfig, ModelConfig
from cnn_chinese_hw.recognizer.model import build_model
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter


_MISSING_MODEL_MSG = (
    "No trained checkpoint found at {path}.\n"
    "Train one first with:\n"
    "    python -m cnn_chinese_hw.recognizer.train\n"
    "(or `--smoke` for a quick pipeline check)."
)


class HandwritingRecognizer:
    def __init__(self, checkpoint_path=None, device='cpu', use_ema=True):
        checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(_MISSING_MODEL_MSG.format(path=checkpoint_path))

        self.device = torch.device(device)
        # The checkpoint contains only tensors + plain containers, so it loads
        # under the safe (weights_only) unpickler.
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        self.classes = list(ckpt['classes'])
        self.data_cfg = DataConfig(**ckpt['data_cfg'])
        model_cfg = ModelConfig(**ckpt['model_cfg'])

        self.model = build_model(model_cfg).to(self.device)
        state = ckpt.get('ema_state') if use_ema else None
        self.model.load_state_dict(state or ckpt['model_state'])
        self.model.eval()

        # Inference is single-instance and may be called from request threads.
        self.lock = threading.Lock()

    def _encode(self, aug, do_augment):
        rastered = aug.raster_strokes(
            image_size=self.data_cfg.image_size, do_augment=do_augment
        )
        x = torch.from_numpy(rastered).permute(2, 0, 1).float().div_(255.0)
        return x.unsqueeze(0).to(self.device)

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
                 Use ``chr(ordinal)`` to get the character.
        """
        aug = HWStrokesAugmenter(strokes_list, find_vertices=True)
        batch = [self._encode(aug, do_augment=False)]
        batch += [self._encode(aug, do_augment=True) for _ in range(max(tta, 0))]
        x = torch.cat(batch, dim=0)

        with self.lock:
            logits = self.model(x)
        probs = F.softmax(logits, dim=1).mean(dim=0)

        n = min(n_cands, probs.numel())
        scores, idx = torch.topk(probs, n)
        return [(float(s), self.classes[int(i)])
                for s, i in zip(scores, idx)]


if __name__ == '__main__':
    rec = HandwritingRecognizer()
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
