<h1> 
  <p align=center> HFTNet: Hierarchical Frequency Transformer Network for Small Drone Detection in Cluttered Scenes </p>
<div align="center">

![Python 3.9](https://img.shields.io/badge/python-3.9-g)
![pytorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-blue.svg)
![TensorRT 8.6](https://img.shields.io/badge/TensorRT-8.6-green.svg)
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)

</div>
</h1>
<img src="assets/fig1.jpg" width="1500">

## Model Zoo 

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>Resolution</th>
      <th>Epoch</th>
      <th>Params(M)</th>
      <th>FLOPs(G)</th>
      <th>$AP$</th>
      <th>$AP_{50}$</th>
      <th>$AP_{75}$</th>
      <th>BaiduYun Download</th>
      <th>Google Download</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>HFTNet</td>
      <td>DUT-Plus</td>
      <td>640</td>
      <td>200</td>
      <td>23.4</td>
      <td>72.3</td>
      <td>62.9</td>
      <td>92.9</td>
      <td>71.4</td>
      <td><a href="https://pan.baidu.com/s/LINK_DUTPLUS">weight</a></td>
      <td>---</td>
    </tr>
    <tr>
      <td>HFTNet</td>
      <td>Det-Fly</td>
      <td>640</td>
      <td>200</td>
      <td>23.4</td>
      <td>72.3</td>
      <td>63.3</td>
      <td>97.1</td>
      <td>72.1</td>
      <td><a href="https://pan.baidu.com/s/LINK_DETFLY">weight</a></td>
      <td>---</td>
    </tr>
  </tbody>
</table>

- Results of the mAP are evaluated on the DUT-Plus dataset (an augmented version of the [DUT-Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV) dataset, available at [DUT-Plus](https://github.com/wanq501/DUT-Plus)) and on the Det-Fly dataset with an input resolution of 640×640.
- All models are trained from scratch without using pretrained weights.

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
      <td>62.9</td>
      <td>60.75</td>
      <td>63.3</td>
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

- Latency is measured on a single NVIDIA RTX 3080Ti GPU with CUDA 12.1 and TensorRT 8.6.
- The INT8 engines are calibrated with the calibration script provided in this repository.

## Code Release

All components are available.

| Component | Status |
| :-- | :-- |
| Model weights for DUT-Plus and Det-Fly (ONNX) | Available |
| Evaluation, test, and detection scripts | Available |
| TensorRT FP16 and INT8 export scripts, including INT8 calibration | Available |
| Dataset splits of DUT-Plus and Det-Fly | Available |
| Source code of HFEN, DIFI, BDFN, and SOIoU, and the training pipeline | Available |

## Dependencies and Installation 

1. Clone and enter the repo.

   ```shell
   git clone https://github.com/wanq501/HFTNet.git
   cd HFTNet
   ```

2. Install dependencies

   ```shell
   pip install -e .
   ```

3. Install the deployment dependencies (only required for TensorRT export)

   ```shell
   pip install onnx onnxsim tensorrt==8.6.1
   ```

## Training and Evaluation 

1. Training

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

- Note: Each script includes detailed instructions on how to set parameters and use the script properly.

## Citation

If you find our repo useful for your research, please cite us:

```
@ARTICLE{HFTNet,
  author={Wan, Qian and Feng, Li and Xiao, Zhiwen and Zhu, Zonghai and Xing, Huanlai and Tian, Yunong and Feng, Yurui and Wei, Zong},
  title={HFTNet: Hierarchical Frequency Transformer Network for Small Drone Detection in Cluttered Scenes}, 
  year={2026},
  note={Under review}}

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