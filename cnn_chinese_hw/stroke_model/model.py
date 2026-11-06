"""Glyph-context transformer: Tomoe strokes -> kai outline representation.

Each stroke is one token: its resampled Tomoe trajectory plus its position in
the stroke order. Self-attention over the whole glyph lets the model infer what
Tomoe discards (true proportions: Tomoe stretches every glyph to its square)
and how neighbouring strokes shape each other. Per stroke it predicts the
representation in ``representation``: centreline points as a residual on the
input trajectory, and non-negative left/right widths.
"""
import torch
from torch import nn

from .representation import N

MAX_STROKES = 40


class StrokeOutlineModel(nn.Module):
    def __init__(self, d_model=256, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N * 2 + N * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.order = nn.Embedding(MAX_STROKES, d_model)
        self.count = nn.Embedding(MAX_STROKES + 1, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.centre = nn.Linear(d_model, N * 2)
        self.widths = nn.Linear(d_model, N * 2)

    def forward(self, inputs, mask):
        """``inputs`` [B, S, N, 2] in Tomoe's 0..1 frame; ``mask`` [B, S] True = stroke.

        Returns centre [B, S, N, 2] and left/right widths [B, S, N] in the
        target frame (1024 units / 1024).
        """
        b, s = inputs.shape[:2]
        deltas = torch.diff(inputs, dim=2, prepend=inputs[:, :, :1])
        x = self.embed(torch.cat([inputs, deltas], dim=-1).reshape(b, s, -1))
        x = x + self.order(torch.arange(s, device=inputs.device))[None]
        x = x + self.count(mask.sum(1).clamp(max=MAX_STROKES))[:, None]
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        centre = inputs + self.centre(x).reshape(b, s, N, 2)
        widths = nn.functional.softplus(self.widths(x)).reshape(b, s, N, 2) * 0.05
        return centre, widths[..., 0], widths[..., 1]
