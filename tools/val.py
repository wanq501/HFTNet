"""Evaluate weights with the protocol used in the paper: batch 1, 640x640, conf 0.001, iou 0.6, max_det 300, FP32,
single-scale, no test-time augmentation.

Usage:
    python tools/val.py --weights runs/train/HFTNet/weights/best.pt --data path/to/data.yaml --split val --device 0
    python tools/val.py --weights a.pt b.pt --data path/to/data.yaml --split test
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)



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
    ap = argparse.ArgumentParser(description='Evaluate HFTNet weights (paper protocol)')
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--device', default='0', help='one GPU index, e.g. 0')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.device       # pin before torch is imported
    from ultralytics import RTDETR
    rows = []
    for w in a.weights:
        m = load_model(w).val(data=a.data, split=a.split, batch=1, imgsz=640, device='0', workers=8, save_json=False,
                          save_hybrid=False, conf=0.001, iou=0.6, max_det=300, half=False, dnn=False, plots=False,
                          verbose=False)
        speed = getattr(m, 'speed', None)
        fps = 1000 / sum(speed.values()) if speed else float('nan')
        rows.append((w, m.box.mp, m.box.mr, m.box.map50, m.box.map75, m.box.map, fps))
    print('\n' + '=' * 96)
    print(f'split={a.split} | batch=1 imgsz=640 conf=0.001 iou=0.6 max_det=300 FP32')
    print(f'{"weights":40} {"P":>6} {"R":>6} {"mAP50":>6} {"mAP75":>6} {"mAP":>6} {"FPS":>6}')
    for w, p, r, m50, m75, m, fps in rows:
        print(f'{w[-40:]:40} {p * 100:6.2f} {r * 100:6.2f} {m50 * 100:6.2f} {m75 * 100:6.2f} {m * 100:6.2f} {fps:6.1f}')
    print('=' * 96)


if __name__ == '__main__':
    main()
