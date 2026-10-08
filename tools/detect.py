"""Run HFTNet on images or a folder and save the drawn results.
Usage: python tools/detect.py --weights best.pt --source images/ --conf 0.25
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
from ultralytics import RTDETR



def load_model(path):
    """RTDETR() accepts only .pt and .yaml files; TensorRT engines and ONNX models are loaded through the base Model
    class with the RT-DETR task map, so they use the RT-DETR validator and predictor."""
    from ultralytics import RTDETR
    if str(path).endswith(('.pt', '.yaml', '.yml')):
        return RTDETR(path)
    from ultralytics.engine.model import Model
    m = RTDETR.__new__(RTDETR)
    Model.__init__(m, model=str(path), task='detect')
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default='runs/detect')
    a = ap.parse_args()
    load_model(a.weights).predict(source=a.source, imgsz=a.imgsz, conf=a.conf, device=a.device, save=True,
                              project=a.project, name='HFTNet', exist_ok=True)


if __name__ == '__main__':
    main()
