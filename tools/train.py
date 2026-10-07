"""Train HFTNet with the recipe of the paper: AdamW, learning rate 1e-4 kept constant after a linear warm-up of
2,000 iterations, weight decay 1e-4, batch 42, 200 epochs, 640x640 input, no mosaic, full precision, seed 0 and
deterministic mode. The optimizer, learning rate, warm-up, weight decay, seed and augmentation come from
ultralytics/cfg/default.yaml of this repository; the values below are the ones set per run.

Usage:
    python tools/train.py --data path/to/data.yaml --name HFTNet-DUT-Plus --device 0
    python tools/train.py --resume runs/train/HFTNet-DUT-Plus/weights/last.pt

Set HFTNET_CKPT=1 to checkpoint the activations of the aggregation nodes (same result, less GPU memory).
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description='Train HFTNet')
    ap.add_argument('--data', default='', help='dataset yaml (required unless --resume is given)')
    ap.add_argument('--model', default=os.path.join(ROOT, 'ultralytics/cfg/models/rt-detr/rtdetr-HFTNet.yaml'))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=42)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help='GPU index, e.g. 0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--project', default='runs/train')
    ap.add_argument('--name', default='HFTNet')
    ap.add_argument('--resume', default='', help='last.pt of an interrupted run')
    ap.add_argument('--box-loss', default='soiou', choices=['soiou', 'mpdiou', 'giou'], help='soiou is HFTNet; the others are for ablations')
    ap.add_argument('--qs-target', default='inner', choices=['inner', 'iou'], help='quality target of query selection')
    a = ap.parse_args()
    if not a.resume and not a.data:
        ap.error('--data is required unless --resume is given')
    os.environ['HFTNET_BOX_LOSS'] = a.box_loss
    os.environ['HFTNET_QS_TARGET'] = a.qs_target
    if a.device.lower() not in ('cpu', 'mps'):
        os.environ['CUDA_VISIBLE_DEVICES'] = a.device      # pin before torch is imported
    from ultralytics import RTDETR
    if a.resume:
        RTDETR(a.resume).train(resume=True)
        return
    RTDETR(a.model).train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device='0' if a.device.lower() not in ('cpu', 'mps') else a.device,
                          workers=a.workers, project=a.project, name=a.name, cache=False, amp=False, mosaic=0.0,
                          cos_lr=False, lrf=1.0, exist_ok=True)


if __name__ == '__main__':
    main()
