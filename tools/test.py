"""Evaluate weights on the test split with the protocol used in the paper: batch 1, 640x640, conf 0.001, iou 0.6,
max_det 300, FP32, single-scale, no test-time augmentation.

Usage:
    python tools/test.py --weights HFTNet_DUT-Plus.pt --data path/to/DUT-Plus.yaml --device 0
"""
import sys

from val import main

if __name__ == '__main__':
    if '--split' not in sys.argv:
        sys.argv += ['--split', 'test']
    main()
