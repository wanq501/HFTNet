"""Export HFTNet to ONNX or build a TensorRT engine.
Usage: python tools/export.py --weights best.pt --format onnx
       python tools/export.py --weights best.pt --format engine --half       TensorRT FP16, the deployment setting of the paper
       python tools/export.py --weights model.onnx --format engine --half    the same, from an exported ONNX file
       python tools/export.py --weights best.pt --int8 --data data.yaml      optional INT8 with entropy calibration
Engines are built with the TensorRT Python API and named <weights>_<precision>.engine. Following standard
mixed-precision practice, FP16 and INT8 engines keep the layer normalization layers in FP32 (--keep-fp32, default
NORMALIZATION), so the FP16 engine retains the full-precision mAP. INT8 calibration uses --calib images (default 500)
sampled evenly from the training split of --data. TensorRT 8.x is required (pip install tensorrt==8.6.1).
"""
import os as _os, sys as _sys


def _pin_gpus(argv):
    """Expose only the requested GPUs before torch is imported. Ultralytics sets CUDA_VISIBLE_DEVICES itself, but too
    late if CUDA was already initialised, which sent a '--device 1' job to GPU 0. The device string passed on to
    Ultralytics stays the same, so its own setting writes the same value again."""
    d = None
    for i, a in enumerate(argv):
        if a == '--device' and i + 1 < len(argv):
            d = argv[i + 1]
            break
        if a.startswith('--device='):
            d = a.split('=', 1)[1]
            break
    if d and d.lower() not in ('cpu', 'mps'):
        _os.environ['CUDA_VISIBLE_DEVICES'] = d


_pin_gpus(_sys.argv)
import argparse, warnings
warnings.filterwarnings('ignore')
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
from ultralytics import RTDETR


def max_onnx_opset():
    """Highest ONNX opset accepted by the installed torch, probed with the same validator torch.onnx.export uses."""
    import importlib
    try:
        from torch.onnx import symbolic_helper
        setter = getattr(symbolic_helper, '_set_opset_version', None)
    except Exception:
        setter = None
    if setter is not None:
        found = None
        for v in range(18, 6, -1):
            try:
                setter(v)
                found = v
                break
            except Exception:
                continue
        try:
            setter(13)
        except Exception:
            pass
        if found:
            return found
    try:
        from torch.onnx import _constants
        for name in ('ONNX_MAX_OPSET', 'onnx_main_opset'):
            if hasattr(_constants, name):
                return int(getattr(_constants, name))
    except Exception:
        pass
    found = None
    for v in range(7, 19):
        try:
            importlib.import_module(f'torch.onnx.symbolic_opset{v}')
            found = v
        except Exception:
            pass
    return found


def grid_sample_exportable():
    """True when torch defines the ONNX symbolic of grid_sampler (opset 16), needed by DCNv3, DySample and the decoder."""
    import importlib
    try:
        return hasattr(importlib.import_module('torch.onnx.symbolic_opset16'), 'grid_sampler')
    except Exception:
        return False


def pick_opset(requested=None):
    """Opset for export: the highest accepted by torch, capped at 17. Raises RuntimeError with the reason if export is impossible."""
    import torch
    top = max_onnx_opset()
    if not top or top < 16 or not grid_sample_exportable():
        raise RuntimeError(f'torch {torch.__version__}: highest ONNX opset {top}, grid_sampler symbolic '
                           f'{"present" if grid_sample_exportable() else "missing"}. grid_sample needs opset 16; '
                           f'export the trained weights in an environment with torch >= 2.0.')
    if requested:
        if requested > top or requested < 16:
            raise RuntimeError(f'opset {requested} not usable with torch {torch.__version__} (accepted range 16..{top})')
        return requested
    return min(top, 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--format', default='onnx', choices=['onnx', 'engine'])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--opset', type=int, default=None, help='default: highest supported by torch, capped at 17')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--device', default='0')
    ap.add_argument('--int8', action='store_true', help='build a TensorRT INT8 engine with entropy calibration')
    ap.add_argument('--data', default='', help='dataset yaml whose training split is used for INT8 calibration')
    ap.add_argument('--calib', type=int, default=500, help='number of calibration images')
    ap.add_argument('--workspace', type=int, default=4, help='TensorRT builder workspace in GB')
    ap.add_argument('--keep-fp32', default='NORMALIZATION', help='TensorRT layer types kept in FP32 in FP16/INT8 engines; empty for none')
    a = ap.parse_args()
    if a.int8 and not a.data:
        ap.error('--int8 needs --data for the calibration images')
    is_onnx = a.weights.lower().endswith('.onnx')
    if a.format == 'onnx' and not a.int8 and is_onnx:
        ap.error('--weights is already an ONNX file')
    onnx_path = a.weights
    if not is_onnx:
        try:
            opset = pick_opset(a.opset)
        except RuntimeError as err:
            raise SystemExit(f'ONNX export not possible: {err}')
        print('ONNX opset:', opset)
        onnx_path = str(RTDETR(a.weights).export(format='onnx', imgsz=a.imgsz, opset=opset, half=False, simplify=True,
                                                 dynamic=False, device=a.device))
        if a.format == 'onnx' and not a.int8:
            print('exported:', onnx_path)
            return
    from trt_utils import split_images, onnx_metadata, build_engine
    precision = 'int8' if a.int8 else ('fp16' if a.half else 'fp32')
    stem = _os.path.splitext(a.weights)[0]
    engine = f'{stem}_{precision}.engine'
    calib, cache = [], None
    if a.int8:
        train = split_images(a.data, 'train'); step = max(1, len(train) // a.calib)
        calib, cache = train[::step][:a.calib], stem + '_int8.cache'
    keep = [t for t in a.keep_fp32.split(',') if t.strip()]
    ver, sec = build_engine(onnx_path, engine, precision, onnx_metadata(onnx_path, a.weights, a.imgsz), a.workspace,
                            calib, cache, a.imgsz, fp32_types=keep)
    print(f'exported: {engine} (TensorRT {ver}, {precision.upper()}, built in {sec:.0f} s)')


if __name__ == '__main__':
    main()
