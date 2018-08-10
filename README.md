This repository is developed by [@penguin1214](https://github.com/penguin1214) and [@Paper99](https://github.com/Paper99).

Our super-resolution framework is built on [BasicSR](https://github.com/xinntao/BasicSR) / [EDSR-Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch) / [RCAN](https://github.com/yulunzhang/RCAN) and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch 0.4.0, CUDA9.0/8.0, cuDNN7/5.1) with NVIDIA GPUs
## Requirements
- Python 3.6
- Pytorch 0.4.0
### Optional
- TensorFlow (for better logging)

## Usage
You should download some SR datasets firstly, for example:
- training datasets: DIV2K (800 images for train, 100 images for validation/test). It can be downloaded from [official site](https://data.vision.ee.ethz.ch/cvl/DIV2K/);
(If you just want to test our SR framework, you can download debug dataset from [here](https://pan.baidu.com/s/1n_iPkVP9GYUcp7Flp_sWrA))
- testing datasets: Set5, Set14, Urban100, B100, Manga109; (The [download link](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo) is provided by [@yulunzhang](https://github.com/yulunzhang))

### Train


 

