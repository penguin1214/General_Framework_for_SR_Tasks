This repository is developed by [@penguin1214](https://github.com/penguin1214) and [@Paper99](https://github.com/Paper99).

Code structure is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [BasicSR](https://github.com/xinntao/BasicSR).

## Requirements
- Python3
- Pytorch
- TensorFlow(for logging, not required.)

## Descriptions:
- 运行train.py所需要的**训练数据集**可在此处 [[link](https://pan.baidu.com/s/1n_iPkVP9GYUcp7Flp_sWrA)] 下载（建议debug时使用），该数据集146M，包含344张LR和HR的样本对。PS: 验证集需自己准备

## Code architecture:
-- data (对数据进行操作)  

-- datasets (存放数据集)  
-- DIV2K  
-- VOC2012

-- models (算法模块)  
models.modules --> modules within networks
    models.modules.blocks --> basic blocks    
    models.modules.xxx_arch --> 特定网络的 building blocks 和 网络的完整结构
models --> solver classes
    models.base_model --> base class solver
    models.networks --> utilitie

-- options (输入参数)  
-- train (训练参数)  
-- test (测试参数)  

-- experiments (存储结果)  
-- train (训练结果)  
-- test （测试结果）

-- scripts (运行脚本)  

-- utils (其它)  

-- tmp_deprecated (暂时弃用的文件)  

-- train.py（训练代码）  

