"""Export a trained checkpoint to ONNX for portable, framework-free inference.

This is the PyTorch-era replacement for the old ``quantize_to_tflite.py``.
ONNX runs under onnxruntime (CPU or GPU) with no TensorFlow dependency, and a
graph can optionally be dynamically quantised to int8 to shrink it.

Usage::

    python -m cnn_chinese_hw.recognizer.export_onnx
    python -m cnn_chinese_hw.recognizer.export_onnx --quantize   # + int8

The class list is written alongside as ``hw_model.classes.json`` so an
onnxruntime-only deployment can map output indices back to ordinals without
importing this package.
"""

import json
import torch
import argparse

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.recognizer.model import build_model
from cnn_chinese_hw.recognizer.config import DataConfig, ModelConfig


def export(checkpoint_path=None, onnx_path=None, use_ema=True, quantize=False):
    checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH
    onnx_path = onnx_path or config.ONNX_PATH

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    data_cfg = DataConfig(**ckpt['data_cfg'])
    model_cfg = ModelConfig(**ckpt['model_cfg'])

    model = build_model(model_cfg)
    model.load_state_dict((ckpt.get('ema_state') if use_ema else None)
                          or ckpt['model_state'])
    model.eval()

    dummy = torch.zeros(1, data_cfg.channels, data_cfg.image_size, data_cfg.image_size)
    # Prefer the modern TorchDynamo-based exporter (the legacy TorchScript path
    # is deprecated in recent PyTorch); fall back if it is unavailable.
    try:
        torch.onnx.export(
            model, (dummy,), onnx_path,
            input_names=['directmap'], output_names=['logits'],
            dynamic_axes={'directmap': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=17, dynamo=True,
        )
    except Exception as e:
        print(f"Dynamo exporter unavailable ({type(e).__name__}); using legacy exporter.")
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=['directmap'], output_names=['logits'],
            dynamic_axes={'directmap': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=17,
        )
    print(f"Exported ONNX -> {onnx_path}")

    classes_path = onnx_path.replace('.onnx', '.classes.json')
    with open(classes_path, 'w') as f:
        json.dump([int(o) for o in ckpt['classes']], f)
    print(f"Wrote class map -> {classes_path}")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise SystemExit("Install onnxruntime to quantize: pip install onnxruntime")
        q_path = onnx_path.replace('.onnx', '.int8.onnx')
        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        print(f"Exported quantized ONNX -> {q_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quantize', action='store_true', help="also emit an int8 graph")
    p.add_argument('--no-ema', action='store_true', help="export raw (non-EMA) weights")
    p.add_argument('--license-group', choices=list(config.LICENSE_GROUPS), default=None,
                   help="export this group's checkpoint (default: legacy hw_model.pt)")
    args = p.parse_args()
    ckpt_path = (config.checkpoint_path_for(args.license_group)
                 if args.license_group else None)
    onnx_path = (config.ONNX_PATH.replace('.onnx', f'.{args.license_group}.onnx')
                 if args.license_group else None)
    export(checkpoint_path=ckpt_path, onnx_path=onnx_path,
           use_ema=not args.no_ema, quantize=args.quantize)


if __name__ == '__main__':
    main()
