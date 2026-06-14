# CNN Handwritten Chinese Recognizer

A **PyTorch** project for **online (stroke-based)** handwritten Chinese /
Japanese-Kanji character recognition. The pen trajectory is rendered into a
small multi-channel "directMap" image and classified by a compact SE-ResNet.
TensorFlow/Keras/TFLite have been fully removed; see
[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the design and its research
citations.

## Project Structure

- **`cnn_chinese_hw/recognizer/`**: All neural-network code.
  - `config.py` — single source of truth for input geometry (48×48×3) and
    hyper-parameters. Import constants from here; never hard-code them.
  - `model.py` — `DirectMapSENet`, the SE-ResNet + global-average-pooling model.
  - `dataset.py` — on-the-fly directMap dataset (replaces the old 1.6 GB `.npz`).
  - `train.py` — training entry point (AdamW, cosine warm-up, label smoothing,
    AMP, EMA, early stopping).
  - `recognizer.py` — `HandwritingRecognizer` inference class.
  - `export_onnx.py` — optional ONNX export (replaces the old TFLite path).
- **`cnn_chinese_hw/client_server/`**: `HWServer`, the service wrapper used by
  `dicts_api`. It lazy-loads the recognizer, so importing it does not require a
  trained checkpoint.
- **`cnn_chinese_hw/stroke_tools/`**: Stroke normalisation, vertex extraction
  and rasterisation (framework-agnostic numpy).
- **`cnn_chinese_hw/parse_data/`**: Tomoe, KanjiVG and Make Me a Hanzi corpus
  parsers. The corpora are license-incompatible (LGPL 2.1 / Arphic PL / CC
  BY-SA), so they are **not** mixed in one model: each `DataConfig.license_group`
  (`config.LICENSE_GROUPS`) trains its own checkpoint — `permissive` (author +
  Tomoe + Make Me a Hanzi, val on held-out KanjiVG) and `ccbysa` (author +
  KanjiVG, val on held-out Tomoe). `recognizer.py` ensembles whatever
  `hw_model.<group>.pt` checkpoints exist by summing calibrated probabilities.
  Train via `train.py --license-group {permissive,ccbysa}`. Licensing/attribution
  rationale: `THIRD_PARTY_DATA.md`.
- **`cnn_chinese_hw/gui/`**: wxPython demo canvases.

## Development & Usage

Use the root virtual environment (`source ../venv/bin/activate`); it already
has `torch` (CUDA) installed.

- **Install**: `pip install -e .` (add `pip install -e ".[onnx]"` / `".[gui]"`
  for optional extras).
- **Smoke-test the pipeline** (seconds):
  `python -m cnn_chinese_hw.recognizer.train --smoke`
- **Train** (writes `data/hw_model.pt`):
  `python -m cnn_chinese_hw.recognizer.train`
- **Inference**: `HandwritingRecognizer().get_candidates_list(strokes)` returns
  `[(score, ordinal), ...]`. The `HWServer` wrapper returns
  `{'cands_list': [...], 'id': ...}`.
- **Tests**: `PYTHONPATH=. pytest tests/ -q` — fast, CPU-only checks of the
  directMap encoding, model forward/backward, training building blocks (GRN,
  EMA warm-up, Mixup/CutMix, SAM) and the inference path + public API contract
  (via a synthetic checkpoint, so no trained model is required).

> **No checkpoint ships in the repo** — recognition (and the `dicts_api`
> handwriting endpoint) needs a model trained first via the command above.

- **Local import**: add the submodule root to `PYTHONPATH` when importing from
  external components (e.g. `dicts_api`).

## Keep this file current

When a change adds, renames, moves, or deprecates a public API (`HandwritingRecognizer`,
`HWServer`, `StrokeData`), a config constant, or a data source covered here, update
this file in the same change. Record only durable, reuse-relevant facts; keep it
terse — no changelogs or one-off detail. See the root `CLAUDE.md` for the convention.
