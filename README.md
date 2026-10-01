<h1>
  <p align=center> HFTNet: Hierarchical Frequency Transformer Network for Small Drone Detection in Cluttered Scenes </p>
<div align="center">

![Python 3.9](https://img.shields.io/badge/python-3.9-g)
![pytorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-blue.svg)
![TensorRT 8.6](https://img.shields.io/badge/TensorRT-8.6-green.svg)
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)

</div>
</h1>
<img src="ultralytics/Assets/fig1.jpg" width="1500">

## Overview

HFTNet is an end-to-end Transformer detector for small drones in cluttered scenes. It decouples local details from global context within each scale and aggregates the feature pyramid along bidirectional pathways, preserving the weak responses of distant drones. Four components address the positions at which the evidence of a distant drone is lost.

- **HFEN** organizes multiscale dilated attention into a five-stage backbone whose sparse sampling matches the spatial extent of distant drones.
- **DIFI** separates high- and low-frequency attention heads within a scale, decoupling local details from global context.
- **BDFN** adaptively aggregates multiscale features along bidirectional pathways.
- **SOIoU** combines an inner-region overlap with a corner-distance penalty, yielding a bounded, size-aware localization signal that also serves as the quality target of query selection.

## Model Zoo

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>Resolution</th>
      <th>Epochs</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$AP_{50}$</th>
      <th>$AP_{75}$</th>
      <th>Weights</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>HFTNet</td>
      <td>DUT-Plus</td>
      <td>640</td>
      <td>200</td>
      <td>21.8</td>
      <td>63.6</td>
      <td>63.2</td>
      <td>93.8</td>
      <td>72.1</td>
      <td><a href="https://github.com/wanq501/HFTNet/releases">ONNX</a></td>
    </tr>
    <tr>
      <td>HFTNet</td>
      <td>Det-Fly</td>
      <td>640</td>
      <td>200</td>
      <td>21.8</td>
      <td>63.6</td>
      <td>60.7</td>
      <td>95.3</td>
      <td>68.2</td>
      <td><a href="https://github.com/wanq501/HFTNet/releases">ONNX</a></td>
    </tr>
  </tbody>
</table>

- All results are reported on the test split of each benchmark with an input resolution of 640×640.
- All models are trained from scratch for 200 epochs without pre-trained weights.

## Datasets

| Dataset | Description | Train / Val / Test |
| :-- | :-- | :-: |
| DUT-Plus | Extends [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV) with multi-target scenes and birds as hard negatives, released with [DQEF-Net](https://github.com/wanq501/DQEF-Net) | 7,000 / 4,000 / 3,000 |
| Det-Fly | Air-to-air images of micro-UAVs ([Zheng et al., IEEE RA-L 2021](https://doi.org/10.1109/LRA.2021.3056059), [dataset](https://github.com/Jake-WU/Det-Fly)), using the split of DQEF-Net | 7,962 / 2,654 / 2,654 |

The split files used in the paper are provided in this repository.

## Deployment

<table>
  <thead align="center">
    <tr>
      <th>Backend</th>
      <th>Precision</th>
      <th>DUT-Plus $AP$</th>
      <th>DUT-Plus Latency (ms)</th>
      <th>Det-Fly $AP$</th>
      <th>Det-Fly Latency (ms)</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>PyTorch</td>
      <td>FP32</td>
      <td>63.2</td>
      <td>60.75</td>
      <td>60.7</td>
      <td>66.49</td>
    </tr>
    <tr>
      <td>TensorRT</td>
      <td>FP16</td>
      <td>63.2</td>
      <td>16.36</td>
      <td>60.6</td>
      <td>18.68</td>
    </tr>
    <tr>
      <td>TensorRT</td>
      <td>INT8</td>
      <td>62.9</td>
      <td>9.51</td>
      <td>60.4</td>
      <td>10.87</td>
    </tr>
  </tbody>
</table>

- Latency is the end-to-end time per image, covering preprocessing, network inference, and post-processing, measured with a batch size of 1 on a single NVIDIA RTX 3080Ti GPU (PyTorch 2.1.0, CUDA 12.1, cuDNN 8.9, TensorRT 8.6).
- The Det-Fly latency includes decoding its 4K images.
- The INT8 engines are calibrated with the calibration script provided in this repository.

## Code Release

The repository is released in two stages.

| Component | Status |
| :-- | :-- |
| Model weights for DUT-Plus and Det-Fly (ONNX) | Available |
| Evaluation, test, and detection scripts | Available |
| TensorRT FP16 and INT8 export scripts, including INT8 calibration | Available |
| Dataset splits of DUT-Plus and Det-Fly | Available |
| Source code of HFEN, DIFI, BDFN, and SOIoU, and the training pipeline | Released upon acceptance |

## Dependencies and Installation

1. Clone and enter the repo.

   ```shell
   git clone https://github.com/wanq501/HFTNet.git
   cd HFTNet
   ```

2. Install dependencies.

   ```shell
   pip install -e .
   ```

3. Install the deployment dependencies (only required for TensorRT export).

   ```shell
   pip install onnx onnxruntime-gpu
   ```

## Training and Evaluation

1. Training (released upon acceptance)

   ```shell
   python tools/train.py
   ```

2. Evaluation

   ```shell
   python tools/val.py
   ```

3. Test

   ```shell
   python tools/test.py
   ```

4. Detect

   ```shell
   python tools/detect.py
   ```

5. Export to TensorRT (FP16 and INT8)

   ```shell
   python tools/export.py
   ```

- Note: Each script includes detailed instructions on how to set its parameters and use it properly.

## Citation

If you find our repo useful for your research, please cite us:

```
@ARTICLE{HFTNet,
  author={Wan, Qian and Feng, Li and Xiao, Zhiwen and Zhu, Zonghai and Xing, Huanlai and Tian, Yunong and Feng, Yurui and Wei, Zong},
  title={HFTNet: Hierarchical Frequency Transformer Network for Small Drone Detection in Cluttered Scenes},
  year={2026},
  note={Under review}
}
```

This project is based on the open source codebase [Ultralytics](https://github.com/ultralytics/ultralytics) and its RT-DETR implementation.

```
@inproceedings{RT-DETR,
  author={Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  title={DETRs Beat YOLOs on Real-time Object Detection},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16965--16974},
  year={2024}
}

@misc{YOLOv8,
  author={Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title={YOLOv8 by Ultralytics},
  version={8.0.0},
  year={2023},
  month={jan},
  license={AGPL-3.0},
  url={https://github.com/ultralytics/ultralytics}
}
```
