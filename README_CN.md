# 4D-A-DSC-LSTM（中文版）

#### 简介

1. 论文题目为：《4D-A-DSC-LSTM: a Spatial–Frequency–Temporal Network based on attention Mechanism for Detecting Driver Fatigue From EEG Signals》

2. 论文原文地址是：

3. 本项目的GITHUB地址是：https://github.com/wangkejie97/4D-A-DSC-LSTM

   

#### 要求

​	在本项目文件夹路径下的终端中使用以下命令（如使用虚拟环境，请先切换到虚拟环境），确保安装需要的依赖包。

```
pip install -r requirements.txt
```

包含了以下五个主要的依赖包。

- numpy==1.19.5
- scikit_learn==1.0.2
- scipy==1.5.4
- torch==1.9.0
- visdom==0.1.8.9



#### 文件以及文件夹功能

- ***DE_3D_Feature.py*** : 将23个被试者的原始脑电数据转换到三维特征。
- ***DE_4D_Feature.py*** : 将三维特征按照二维地形图（参照论文）转换成为四维特征。
- ***dataloader*** : 将四维特征与数据集标签，按照自定义的五折交叉验证，划分为训练集（4/5）与测试集（1/5）。
- ***train*** : 训练与测试，可通过visdom在网页端实时显示训练曲线。
- ***myNet*** : 定义的4D-A-DSC-LSTM模型。
- ***"./processedData/"*** : 用于存放转换后的三维特征与四维特征。
- ***"./pth/"*** : 用于存放第n折训练中的最高准确率的模型。



#### 快速使用步骤

1. 打开"4D-A-DSC-LSTM/DE_3D_Feature"，在第92行处，替换至你电脑中实际的数据集路径，然后运行该py文件，完成后将会在"4D-A-DSC-LSTM/processedData"下生成"data_3d.npy"文件。

2. 打开"4D-A-DSC-LSTM/DE_4D_Feature"，直接运行，完成后将会在"4D-A-DSC-LSTM/processedData"下生成"data_4d.npy"文件。

3. 打开"4D-A-DSC-LSTM/dataloader"，可调整使用五折交叉验证中的第几折进行验证，设置batch_size，或者设置随机数种子。

4. 打开"4D-A-DSC-LSTM/train"，在训练开始前，请先打开cmd命令行，（如使用虚拟环境，请先切换），输入

   ```
   python -m visdom.server
   ```

然后，打开提示中的网站进行实时可视化。可以自行调整学习率或Epoch等。



##### 其他

- 注意力可视化可通过模型网络输出的spaAtten, freqAtten得到。
- 需要在训练时打开visdom。
