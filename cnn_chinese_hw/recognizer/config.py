"""Single source of truth for the recognizer's hyper-parameters and paths.

Historically the input geometry was duplicated (and *disagreed*: the old
Keras model used a 20x20 grid while the TFLite recognizer assumed 28x28).
Everything that touches the model now imports from here so the encoder,
trainer and recognizer can never drift apart again.
"""

from dataclasses import dataclass, field, asdict
from cnn_chinese_hw.get_package_dir import get_package_dir


# ---------------------------------------------------------------------------
# Input representation
# ---------------------------------------------------------------------------
# The character is rendered into a small multi-channel "directMap"-style
# raster (see docs/ARCHITECTURE.md). Each channel encodes a different aspect
# of the online stroke trajectory:
#   ch0 - intensity weighted towards the *start* of every stroke
#   ch1 - intensity weighted towards the *end* of every stroke
#   ch2 - intensity weighted towards the earliest *strokes* (stroke order)
# Together these make stroke direction and stroke order recoverable from a
# single fixed-size image, which is what lets the model tolerate
# out-of-order drawing.
IMAGE_SIZE = 48
CHANNELS = 3

# Normalised internal coordinate space the strokes are scaled into before
# rasterising (matches the Tomoe data's native 0..1000 space).
WRITING_BOX = 1000


# ---------------------------------------------------------------------------
# Data / augmentation
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    image_size: int = IMAGE_SIZE
    channels: int = CHANNELS
    # On-the-fly augmentation replaces the old pre-expanded 1.6 GB .npz cache.
    # Each epoch every base sample is re-augmented, so these are now "virtual
    # copies per base sample within one epoch" rather than a fixed dataset
    # blow-up on disk.
    augmentations_per_sample: int = 8
    real_copies_per_sample: int = 1
    # Restrict to a handful of classes for fast smoke-testing of the pipeline.
    small_sample_only: bool = False
    small_sample_size: int = 200
    # Cache the *parsed stroke lists* (small) rather than rendered rasters.
    cache_strokes: bool = True
    num_workers: int = 8


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    in_channels: int = CHANNELS
    # Channels per stage of the SE-residual backbone.
    stage_channels: tuple = (64, 128, 256)
    # Residual blocks per stage.
    blocks_per_stage: tuple = (2, 2, 2)
    stem_channels: int = 48
    se_reduction: int = 8
    drop_rate: float = 0.2          # classifier dropout
    drop_path_rate: float = 0.1     # stochastic depth (max, linearly scaled)
    # Anti-aliased downsampling (blur-before-subsample). Improves shift
    # invariance, which matters because the character's position in the grid
    # varies (Zhang, "Making Convolutional Networks Shift-Invariant Again",
    # ICML 2019).
    anti_alias: bool = True
    blur_filt_size: int = 3
    num_classes: int = 0            # filled in at train time from the data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 256
    lr: float = 3e-3                # AdamW peak LR
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    amp: bool = True                # mixed precision
    ema_decay: float = 0.999        # exponential moving average of weights
    # Ramp the EMA decay up early on (decay ~ (1+step)/(10+step) capped at
    # ema_decay). Without this the averaged model lags far behind a rapidly
    # improving net at the start of training (Tarvainen & Valpola, "Mean
    # teachers are better role models", NeurIPS 2017).
    ema_warmup: bool = True
    early_stop_patience: int = 25   # epochs without val-acc improvement
    seed: int = 1234
    # Mixup / CutMix input mixing (the main regularisers in the modern
    # "ResNet strikes back" recipe, Wightman et al., arXiv:2110.00476, 2021).
    # 0 disables. For a full run try mixup_alpha=0.1 and cutmix_alpha=1.0;
    # left off by default because the benefit on small, fine-grained,
    # sparse-stroke data is recipe-dependent and best confirmed empirically.
    mixup_alpha: float = 0.0        # Beta(a, a); Zhang et al., ICLR 2018
    cutmix_alpha: float = 0.0       # Beta(a, a); Yun et al., ICCV 2019
    mix_prob: float = 0.5           # chance a batch is mixed when enabled
    # An IME presents a *candidate list*, so top-k accuracy is the metric we
    # select/early-stop on, not top-1 (top-k convention: Russakovsky et al.,
    # "ImageNet Large Scale Visual Recognition Challenge", IJCV 2015).
    topk: int = 10
    # Re-weight sampling by the inverse effective number of samples per class
    # to counter the class imbalance from multi-script / multi-drawing chars
    # (Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
    # CVPR 2019).
    balanced_sampling: bool = True
    cb_beta: float = 0.999


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
_DATA = f"{get_package_dir()}/data"

# Trained checkpoint (state_dict + config + class list, all in one file).
CHECKPOINT_PATH = f"{_DATA}/hw_model.pt"
# Parsed-stroke cache (small; ~tens of MB, regenerated if missing).
STROKE_CACHE_PATH = f"{_DATA}/stroke_cache.pkl"
# Optional exported inference graph.
ONNX_PATH = f"{_DATA}/hw_model.onnx"

# Source corpora (unchanged from the original project).
TOMOE_FILES = {
    "ja": f"{_DATA}/handwriting-ja.xml",
    "zh": f"{_DATA}/handwriting-zh_CN.xml",
    "zh-TW": f"{_DATA}/handwriting-zh_TW.xml",
}
SUPPLEMENTAL_PATH = f"{_DATA}/supplemental_data.json"
KANJIVG_PATH = f"{_DATA}/validation/kanjivg.xml"


def default_configs():
    """Convenience bundle used by the training entry point."""
    return DataConfig(), ModelConfig(), TrainConfig()


def config_to_dict(cfg) -> dict:
    return asdict(cfg)
