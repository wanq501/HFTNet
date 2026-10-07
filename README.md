<h1 align="center">HFTNet: Hierarchical Frequency Transformer Network for Small Drone Detection in Cluttered Scenes</h1>

<div align="center">

![Python 3.9](https://img.shields.io/badge/python-3.9-green) ![PyTorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-orange) ![License AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

</div>

This repository contains the code, configuration and trained weights of HFTNet, a real-time DETR detector for small drones in cluttered scenes. It is built on the RT-DETR implementation of Ultralytics.

## Method

HFTNet adds four components to RT-DETR-R18:

| Component | Role | Code |
|---|---|---|
| **HFEN** | Hierarchical feature extraction backbone with multiscale dilated attention blocks | `ultralytics/nn/Addmodules/MSDA.py` (`BasicBlock_MSDA`) |
| **DIFI** | Dual-frequency intra-scale interaction on the stride-32 map: global multi-head attention captures low-frequency context, and a 2x2 window-attention branch adds high-frequency detail through a channel-wise gate initialised to zero | `ultralytics/nn/Addmodules/DIFI.py` (`AIFI_DF`) |
| **BDFN** | Bidirectional dynamic fusion network: DySample up-sampling, learnable weighted fusion and a DFAL at every fusion node. DFAL aggregates the node input through two cascaded DFCS stages and refines it with a deformable feature bottleneck (DFBN, DCNv3 sampling) | `ultralytics/nn/Addmodules/Dysample.py`, `BiFPN.py`, `BDFN.py` (`DFAL2`, `DFBN`, `DCNv3`) |
| **SOIoU** | Box regression loss that combines the overlap of inner boxes with corner-distance penalties normalised by the image diagonal. The inner-box IoU is also the quality target of query selection | `ultralytics/utils/metrics.py` (`soiou`, `inner_iou`), used in `ultralytics/models/utils/loss.py` |

For a predicted box and its matched ground-truth box,

$$\mathcal{L}_{\mathrm{SOIoU}} = 1-\mathrm{IoU}^{\mathrm{inner}}+\frac{d_1^{2}+d_2^{2}}{W^{2}+H^{2}},$$

where $\mathrm{IoU}^{\mathrm{inner}}$ is the IoU of the two inner boxes, which keep the box centres and scale the sides by $r=0.75$, $d_1$ and $d_2$ are the distances between the top-left and between the bottom-right corners of the two boxes, and $W\times H$ is the input size.

The model configuration is `ultralytics/cfg/models/rt-detr/rtdetr-HFTNet.yaml`.

## Results

| Dataset | Split | mAP | mAP50 | mAP75 | Precision | Recall | Params | FLOPs | Weights |
|---|---|---|---|---|---|---|---|---|---|
| DUT-Plus | val | 62.91 | 92.92 | 71.36 | 97.06 | 89.84 | 23.4M | 72.3G | [HFTNet_DUT-Plus.pt](https://github.com/wanq501/HFTNet/releases) |
| Det-Fly | val | 63.25 | 97.13 | 72.11 | 98.29 | 95.48 | 23.4M | 72.3G | [HFTNet_Det-Fly.pt](https://github.com/wanq501/HFTNet/releases) |

Evaluation protocol: 640x640, batch 1, confidence threshold 0.001, IoU threshold 0.6, max 300 detections, FP32, single-scale, no test-time augmentation (`tools/val.py`).

## Installation

Tested with Python 3.9 and PyTorch 1.12.1.

```bash
conda create -n hftnet python=3.9 -y && conda activate hftnet
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Data

Prepare each dataset in YOLO format and describe it in a yaml file:

```yaml
path: /data/DUT-Plus
train: images/train
val: images/val
test: images/test
nc: 1
names: [UAV]
```

The image lists of the splits used in the paper are in `splits/`. They were written by `tools/export_splits.py`.

## Training

```bash
python tools/train.py --data path/to/DUT-Plus.yaml --name HFTNet-DUT-Plus --device 0
python tools/train.py --data path/to/Det-Fly.yaml --name HFTNet-Det-Fly --device 0
```

The recipe follows the paper: AdamW, learning rate 1e-4 kept constant after a linear warm-up of 2,000 iterations, weight decay 1e-4, batch 42, 200 epochs, no mosaic, full precision, seed 0 and deterministic mode. Batch 42 fits one 80 GB GPU. Setting `HFTNET_CKPT=1` checkpoints the activations of the aggregation nodes, which lowers the memory with the same result. An interrupted run continues with `--resume runs/train/<name>/weights/last.pt`.

`--box-loss giou` and `--qs-target iou` restore the RT-DETR loss and query-selection target for ablations.

## Evaluation, inference and export

```bash
# paper protocol
python tools/val.py --weights HFTNet_DUT-Plus.pt --data path/to/DUT-Plus.yaml --split val --device 0
# detection on images or a folder
python tools/detect.py --weights HFTNet_DUT-Plus.pt --source path/to/images --device 0
# ONNX, or a TensorRT FP16 engine
python tools/export.py --weights HFTNet_DUT-Plus.pt --format onnx
python tools/export.py --weights HFTNet_DUT-Plus.pt --format engine --half
```

## Citation

The citation will be added when the paper is published.

## License and acknowledgements

This project is released under the AGPL-3.0 license, following Ultralytics. It builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR), and uses the window attention of Swin Transformer, DySample, and DCNv3.
