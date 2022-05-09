# 4D-A-DSC-LSTM代码使用方法（中文版）

### 简介

1. 论文题目为：《》
2. 论文原文地址是：
3. 本文用到的框架为pytorch，有部分依赖包请在代码中导入，我们在训练测试代码中使用visdom进行实时可视化，需要自行安装与查看使用。

#### 使用步骤

1. 打开"4D-A-DSC-LSTM/DE_3D_Feature"，在第83行处，将你的SEED-VIG中原始脑电信号的文件夹路径替换进去，直接运行，直到计算结束后，在"4D-A-DSC-LSTM/processedData"下产生一个叫"data_3d.npy"的文件。
2. 打开"4D-A-DSC-LSTM/DE_4D_Feature"，直接运行，在"4D-A-DSC-LSTM/processedData"下产生一个叫"data_4d.npy"的文件。
3. 打开"4D-A-DSC-LSTM/dataloader"，在第7行可以调整使用5折交叉验证中的第几折测试，第9行可以设置batch_size，第11行可以设置随机数种子。
4. 打开"4D-A-DSC-LSTM/train"，第13行的acc_low，表示测试集准确率如果高于它，则保存为.pth；第21行调整学习率；第24行可以调整Epoch；注意，在训练开始前，务必打开你的visdom进行实时检测。
