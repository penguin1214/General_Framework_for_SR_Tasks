## Notice
- 每次更改.json文件时别忘了更改options.py以及train.py文件（**Most Important**）

## Descriptions
**name**:你自己定义的名字
**exec_debug**： 表式执行的时候debug
**net_debug**: 寻找一个合适的超参数
**mode**: 'sr'表示只有生成器，‘srgan’表示有生成器和判别器
**scale**: 上采样的倍数
**datasets**： 

分为两种phase： 

train -- 训练集图像 
- name: 所使用的数据集名字，方便log 
- mode: 读取文件夹的方式。'LRHR'：从低分辨率图像文件夹和高分辨率图像文件夹中读取LR-HR的样本对 
- dataroot_HR： 指定欲读取的高分辨率图像文件夹 
- dataroot_LR： 指定欲读取的低分辨率图像文件夹 
- n_workers： 读取图像数据的线程数 
- batch_size： 每次Iteration送入的样本数量 
- HR_size: 指定高分辨率图像的patch大小 
- use_flip: 是否翻折图像 
- use_rot: 是否旋转图像 
- noise: 给低分辨率图像加何种噪声。<'.'>不加噪声；<'G'>加入高斯噪声；<'S'>加入泊松噪声 

val -- 验证集图像
- name: 所使用的数据集名字，方便log 
- mode: 读取文件夹的方式。'LRHR'：从低分辨率图像文件夹和高分辨率图像文件夹中读取LR-HR的样本对 
- dataroot_HR： 指定欲读取的高分辨率图像文件夹 
- dataroot_LR： 指定欲读取的低分辨率图像文件夹 

**networks**： 
