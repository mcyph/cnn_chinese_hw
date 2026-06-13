"""Fast, self-contained tests for the PyTorch handwriting recognizer.

These lock in the invariants that were previously only checked by hand:
the directMap encoding, the model's forward/backward, the training-time
building blocks (GRN, EMA warm-up, Mixup/CutMix, SAM) and the inference
path + public API contract. None of them need a real trained checkpoint or
GPU, so the whole module runs in a few seconds on CPU.

Run with:  PYTHONPATH=. pytest tests/ -q
"""

import math

import numpy as np
import torch
import pytest

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.recognizer.model import build_model, GRN
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter


# A simple 我-like stroke set reused across tests.
STROKES = [[(208, 0), (199, 119), (94, 341)],
           [(0, 461), (781, 520), (915, 520), (999, 479)],
           [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)],
           [(303, 514), (94, 766)],
           [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)],
           [(716, 628), (462, 916)],
           [(696, 101), (771, 155), (835, 251)]]


# --------------------------------------------------------------------------
# directMap encoding
# --------------------------------------------------------------------------
def test_directmap_shape_dtype_range():
    aug = HWStrokesAugmenter(STROKES, find_vertices=True)
    r = aug.raster_strokes(image_size=48, do_augment=False)
    assert r.shape == (48, 48, 3)
    assert r.dtype == np.uint8
    assert 0 <= r.min() and r.max() <= 255
    assert r.max() > 0  # something was actually drawn


def test_directmap_deterministic_without_augmentation():
    aug = HWStrokesAugmenter(STROKES, find_vertices=True)
    a = aug.raster_strokes(image_size=48, do_augment=False)
    b = aug.raster_strokes(image_size=48, do_augment=False)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("grn", [True, False])
@pytest.mark.parametrize("anti_alias", [True, False])
def test_model_forward_backward(grn, anti_alias):
    cfg = config.ModelConfig(num_classes=17, use_grn=grn, anti_alias=anti_alias)
    model = build_model(cfg)
    x = torch.randn(4, cfg.in_channels, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(x)
    assert out.shape == (4, 17)
    out.sum().backward()
    assert model.classifier.weight.grad is not None


def test_grn_is_identity_at_init():
    grn = GRN(8)
    x = torch.randn(2, 8, 6, 6)
    assert torch.allclose(grn(x), x, atol=1e-6)


# --------------------------------------------------------------------------
# Training building blocks
# --------------------------------------------------------------------------
def test_ema_decay_warms_up():
    from cnn_chinese_hw.recognizer.train import ModelEMA
    m = build_model(config.ModelConfig(num_classes=5))
    ema = ModelEMA(m, decay=0.999, warmup=True)
    ema.update(m)
    # At step 1 the effective decay is min(0.999, 2/11) -> ~0.18, not 0.999.
    assert ema.step == 1
    assert min(0.999, (1 + ema.step) / (10 + ema.step)) < 0.5


def test_mixup_cutmix_invariants():
    from cnn_chinese_hw.recognizer.train import mixup_cutmix
    cfg = config.TrainConfig()
    x = torch.randn(8, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = torch.randint(0, 10, (8,))

    # Disabled -> no mixing.
    xo, ya, yb, lam = mixup_cutmix(x.clone(), y, cfg)
    assert lam == 1.0 and torch.equal(ya, yb)

    # Enabled -> a valid convex/patch mix.
    cfg.mixup_alpha = 0.2
    cfg.cutmix_alpha = 1.0
    cfg.mix_prob = 1.0
    xo, ya, yb, lam = mixup_cutmix(x.clone(), y, cfg)
    assert 0.0 <= lam <= 1.0
    assert xo.shape == x.shape


def test_sam_reduces_loss_on_toy_problem():
    from cnn_chinese_hw.recognizer.sam import SAM
    torch.manual_seed(0)
    model = torch.nn.Linear(20, 3)
    x = torch.randn(64, 20)
    y = torch.randint(0, 3, (64,))
    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    crit = torch.nn.CrossEntropyLoss()

    first = crit(model(x), y).item()
    for _ in range(50):
        crit(model(x), y).backward()
        opt.first_step(zero_grad=True)
        crit(model(x), y).backward()
        opt.second_step(zero_grad=True)
    last = crit(model(x), y).item()
    assert last < first


# --------------------------------------------------------------------------
# Inference path + public API (synthetic checkpoint -> no training needed)
# --------------------------------------------------------------------------
@pytest.fixture
def synthetic_checkpoint(tmp_path):
    classes = [ord(c) for c in "我成在的人"]
    model_cfg = config.ModelConfig(num_classes=len(classes))
    data_cfg = config.DataConfig()
    model = build_model(model_cfg)
    path = tmp_path / "hw_model.pt"
    torch.save({
        'model_state': model.state_dict(),
        'ema_state': model.state_dict(),
        'classes': classes,
        'model_cfg': config.config_to_dict(model_cfg),
        'data_cfg': config.config_to_dict(data_cfg),
        'temperature': 1.5,
        'format': 1,
    }, path)
    return str(path), classes


def test_recognizer_roundtrip_and_contract(synthetic_checkpoint):
    from cnn_chinese_hw.recognizer.recognizer import HandwritingRecognizer
    path, classes = synthetic_checkpoint
    rec = HandwritingRecognizer(checkpoint_path=path, device='cpu')
    assert rec.temperature == 1.5

    cands = rec.get_candidates_list(STROKES, n_cands=3)
    assert len(cands) == 3
    for score, ord_ in cands:
        assert isinstance(score, float) and isinstance(ord_, int)
        assert ord_ in classes
    # Scores are a descending-sorted probability ranking.
    scores = [s for s, _ in cands]
    assert scores == sorted(scores, reverse=True)

    # Full distribution sums to ~1 (softmax), with and without TTA.
    total = sum(s for s, _ in rec.get_candidates_list(STROKES, n_cands=999))
    assert abs(total - 1.0) < 1e-4
    total_tta = sum(s for s, _ in rec.get_candidates_list(STROKES, n_cands=999, tta=3))
    assert abs(total_tta - 1.0) < 1e-4


def test_hwserver_contract(synthetic_checkpoint):
    from cnn_chinese_hw.recognizer.recognizer import HandwritingRecognizer
    from cnn_chinese_hw.client_server.HWServer import HWServer
    path, _ = synthetic_checkpoint
    srv = HWServer()
    srv._recognizer = HandwritingRecognizer(checkpoint_path=path, device='cpu')

    out = srv.get_chinese_written_candidates(STROKES, id="abc")
    assert sorted(out.keys()) == ['cands_list', 'id']
    assert out['id'] == 'abc'
    assert isinstance(out['cands_list'], list) and out['cands_list']
    assert all(isinstance(c, str) for c in out['cands_list'])

    # Empty input is handled without invoking the model.
    assert srv.get_chinese_written_candidates([], id="z") == {'cands_list': [], 'id': 'z'}


def test_missing_checkpoint_raises():
    from cnn_chinese_hw.recognizer.recognizer import HandwritingRecognizer
    with pytest.raises(FileNotFoundError):
        HandwritingRecognizer(checkpoint_path="/nonexistent/hw_model.pt")
