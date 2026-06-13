"""DirectMap SE-ResNet: a compact modern CNN for online HCCR.

Design rationale (full discussion and citations in docs/ARCHITECTURE.md):

* The input is a multi-channel *directMap*-style raster of the online stroke
  trajectory. Feeding a normalisation-cooperated directional feature map into
  a deep CNN is the representation that set the online/offline HCCR benchmark
  in Zhang, Bengio & Liu, "Online and Offline Handwritten Chinese Character
  Recognition: A Comprehensive Study and New Benchmark", Pattern Recognition
  61 (2017) 348-360 (arXiv:1606.05763).

* The backbone is a residual network (He et al., "Deep Residual Learning for
  Image Recognition", CVPR 2016) with Squeeze-and-Excitation channel attention
  (Hu, Shen & Sun, "Squeeze-and-Excitation Networks", CVPR 2018) and a few
  ConvNeXt-era micro-design choices -- GELU activations and a single
  normalisation/activation per residual function (Liu et al., "A ConvNet for
  the 2020s", CVPR 2022, arXiv:2201.03545) -- plus Global Response Normalization
  to counter channel feature collapse (Woo et al., "ConvNeXt V2", CVPR 2023,
  arXiv:2301.00808).

* The huge flatten + 1024/512 fully-connected head of the original Keras model
  is replaced by global average pooling, which removes the vast majority of the
  parameters at no accuracy cost. This is the standard compactness trick for
  HCCR (Xiao et al., "Building Fast and Compact Convolutional Neural Networks
  for Offline Handwritten Chinese Character Recognition", arXiv:1702.07975).

* Stochastic depth (Huang et al., "Deep Networks with Stochastic Depth",
  ECCV 2016) regularises the residual stack.

* Downsampling is anti-aliased: a low-pass blur is applied before
  subsampling, which restores the shift-invariance that plain strided
  convolutions break. The character's position within the grid varies from
  sample to sample, so this is directly useful here (Zhang, "Making
  Convolutional Networks Shift-Invariant Again", ICML 2019, arXiv:1904.11486).

The result is on the order of a couple of million parameters -- roughly an
order of magnitude smaller than the original >30M-parameter dense model -- and
is trained from scratch with the recipe in train.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_chinese_hw.recognizer.config import ModelConfig


class BlurPool2d(nn.Module):
    """Anti-aliased subsampling: depthwise low-pass (binomial) blur followed by
    strided sampling (Zhang, ICML 2019). A drop-in replacement for a stride-2
    operation that does not alias the signal."""

    def __init__(self, channels: int, stride: int = 2, filt_size: int = 3):
        super().__init__()
        self.stride = stride
        self.pad = filt_size // 2
        # Binomial coefficients give a separable low-pass filter, e.g.
        # filt_size=3 -> [1, 2, 1]; outer product -> 2-D kernel; normalise.
        coeffs = torch.tensor(
            [math.comb(filt_size - 1, i) for i in range(filt_size)],
            dtype=torch.float32,
        )
        kernel = coeffs[:, None] * coeffs[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer(
            "kernel", kernel[None, None].repeat(channels, 1, 1, 1)
        )
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.channels)


class GRN(nn.Module):
    """Global Response Normalization (Woo et al., "ConvNeXt V2", CVPR 2023,
    arXiv:2301.00808). Normalises each channel by its global (spatial) L2 norm
    relative to the mean across channels, then calibrates. This increases
    inter-channel competition and counters the feature-collapse / redundant-
    activation behaviour of plain ConvNets. With ``gamma``/``beta`` initialised
    to zero the layer is the identity at start of training, so it is safe to
    enable by default."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)        # (N, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)      # channel competition
        return self.gamma * (x * nx) + self.beta + x


class SqueezeExcite(nn.Module):
    """Channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)  # global average pool (squeeze)
        s = self.fc2(self.act(self.fc1(s)))   # excitation
        return x * self.gate(s)


class DropPath(nn.Module):
    """Per-sample stochastic depth (Huang et al., ECCV 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * mask.floor_() / keep


class SEResidualBlock(nn.Module):
    """Pre-activation residual block: BN-GELU-Conv x2 + SE, with a projection
    shortcut when the channel count or spatial resolution changes.

    When ``anti_alias`` is set, the stride-2 downsampling is performed by a
    stride-1 convolution followed by a :class:`BlurPool2d` (Zhang, ICML 2019)
    on both the residual and the shortcut paths, rather than by an aliasing
    strided convolution.
    """

    def __init__(self, in_ch, out_ch, stride=1, se_reduction=8, drop_path=0.0,
                 anti_alias=True, blur_filt_size=3, use_grn=True):
        super().__init__()
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.grn = GRN(out_ch) if use_grn else nn.Identity()
        self.se = SqueezeExcite(out_ch, se_reduction)
        self.drop_path = DropPath(drop_path)

        downsample = stride != 1
        # If anti-aliasing, the conv stays stride-1 and a BlurPool does the
        # subsampling; otherwise the conv itself strides.
        conv_stride = 1 if (not downsample or anti_alias) else stride
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=conv_stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        if downsample and anti_alias:
            self.aa_main = BlurPool2d(out_ch, stride, blur_filt_size)
        else:
            self.aa_main = nn.Identity()

        if downsample or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=conv_stride, bias=False)
            self.aa_short = (BlurPool2d(out_ch, stride, blur_filt_size)
                             if downsample and anti_alias else nn.Identity())
        else:
            self.shortcut = None  # identity shortcut (no downsample, same ch)
            self.aa_short = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(x))
        if self.shortcut is None:
            identity = x
        else:
            identity = self.aa_short(self.shortcut(out))
        out = self.aa_main(self.conv1(out))
        out = self.conv2(self.act(self.bn2(out)))
        out = self.grn(out)
        out = self.se(out)
        return identity + self.drop_path(out)


class DirectMapSENet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.num_classes > 0, "ModelConfig.num_classes must be set"
        self.cfg = cfg

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.stem_channels),
            nn.GELU(),
        )

        # Linearly increasing drop-path rate across the whole residual stack.
        total_blocks = sum(cfg.blocks_per_stage)
        dpr = [cfg.drop_path_rate * i / max(total_blocks - 1, 1)
               for i in range(total_blocks)]

        blocks = []
        in_ch = cfg.stem_channels
        block_idx = 0
        for stage, (out_ch, n_blocks) in enumerate(
                zip(cfg.stage_channels, cfg.blocks_per_stage)):
            for b in range(n_blocks):
                # Downsample (stride 2) on the first block of every stage.
                stride = 2 if b == 0 else 1
                blocks.append(SEResidualBlock(
                    in_ch, out_ch, stride=stride,
                    se_reduction=cfg.se_reduction,
                    drop_path=dpr[block_idx],
                    anti_alias=cfg.anti_alias,
                    blur_filt_size=cfg.blur_filt_size,
                    use_grn=cfg.use_grn,
                ))
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        self.head_norm = nn.BatchNorm2d(in_ch)
        self.head_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.classifier = nn.Linear(in_ch, cfg.num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_act(self.head_norm(x))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig) -> DirectMapSENet:
    return DirectMapSENet(cfg)
