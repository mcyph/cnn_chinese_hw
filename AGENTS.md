# CNN Handwritten Chinese Recognizer

A **PyTorch** project for **online (stroke-based)** handwritten Chinese /
Japanese-Kanji character recognition. The pen trajectory is rendered into a
small multi-channel "directMap" image and classified by a compact SE-ResNet.
TensorFlow/Keras/TFLite have been fully removed; see
[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the design and its research
citations.

## Capability → canonical entrypoint

| Capability | Canonical entrypoint |
|---|---|
| Calibrated ensemble candidates | `recognizer.recognizer.HandwritingRecognizer.get_candidates_list` |
| Native per-checkpoint logits, vocabularies and calibration metadata | `HandwritingRecognizer.infer_logits`; shared Pydantic evidence and before/after construction fingerprints from `iso_tools.inference.artifacts.LoadIdentity` (available on the monorepo `PYTHONPATH`); resident identity never follows a subsequently replaced checkpoint |
| Import-light native probability ensemble and compact diagnostic evidence | `recognizer.evidence.project_handwriting`; checkpoint temperatures preserve existing ensemble semantics, missing classes are explicit and probabilities are not writer-calibrated correctness |
| Separate checkpoint selection, temperature fitting and corpus testing | `recognizer.calibration.split_calibration_samples`, using `iso_tools.stt.evaluation.partitions`; protects held-out character variants/rounded geometry and excludes exact training duplicates; writer/device identities remain unavailable |
| Stroke outline model (Tomoe trajectories → kai outlines) | `stroke_model.representation` (centreline + left/right widths at `N` cosine-spaced samples; `encode` from an outline+median, `outline_path` back to SVG, raster `glyph_iou`), `stroke_model.dataset` (per-axis-normalised Tomoe ↔ AnimCJK/Make Me a Hanzi pair matcher, character-hash splits, `pairs.npz`), `stroke_model.synthetic` (Tomoe-style inputs from train-split outline medians, calibrated against real pairs; `--measure`), `stroke_model.model.StrokeOutlineModel`, `stroke_model.train` (`--synthetic`; held-out raster IoU vs a box-aligned uniform-width baseline), `stroke_model.infer` |
| Lazy service wrapper | `client_server.HWServer`; synchronized per-instance first load, retryable after failure; backend adapters must reuse one wrapper |
| Character stroke display | `parse_data.display_strokes.display_strokes` returns one source per locale with its coordinate transform: `zh` = Make Me a Hanzi (unscreened: it follows PRC order where Tomoe zh does not) → Tomoe; `zh-TW` = AnimCJK Taiwan → Make Me a Hanzi screened by `_compatible` against Tomoe zh-TW → Tomoe; `ja` = AnimCJK Japanese where `_agrees_with_kanjivg` (count + direction) → KanjiVG (`parse_data.kanjivg_strokes.kanjivg_strokes`: sampled Bézier centrelines in the 1000-unit frame plus each stroke's CJK stroke type as `kind`) → screened Make Me a Hanzi → Tomoe. Outline corpora are optional, git-ignored and indexed lazily: Make Me a Hanzi via `iter_makemeahanzi_data --download`, AnimCJK (pinned commit + sha256, kana file deliberately excluded) via `parse_data.animcjk_data --download`. Screens are not certification of a regional standard. Wherever only Tomoe remains, `stroke_model.infer.generated_strokes` (`source: 'generated'`, cached per character, CPU, lazily loaded checkpoint `data/stroke_model/stroke_outline.pt`) draws kai outlines from Tomoe's strokes in Tomoe's order; no checkpoint = Tomoe. The frontend draws every centreline-only source with the brush renderer `dicts_api/frontend/src/components/animations/HanziAnimation/brush.ts`. |

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
- **`cnn_chinese_hw/stroke_model/`**: display-only model that turns Tomoe
  strokes into kai outlines for characters no outline source covers (see
  README "Character stroke display data"). Not part of recognition.
- **`cnn_chinese_hw/gui/`**: wxPython demo canvases.

## Development & Usage

Use the root virtual environment (`source ../venv/bin/activate`); it already
has `torch` (CUDA) installed.

- **Install**: `pip install -e .` (add `pip install -e ".[onnx]"` / `".[gui]"`
  for optional extras).
- **Smoke-test the pipeline** (seconds; writes `data/hw_model.<group>.smoke.pt`,
  never the served checkpoint):
  `python -m cnn_chinese_hw.recognizer.train --smoke`
- **Train** (writes `data/hw_model.<group>.pt`, default group `permissive`):
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
terse — no changelogs or one-off detail. See the root `AGENTS.md` for the convention.
