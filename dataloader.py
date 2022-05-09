import torch
import numpy as np
from toolbox import myDataset_5cv


#  choose n fold
n = 0
# batch_size
batch_size = 150
# 随机数种子
seed = 20
# 使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### 1.数据集 ###
data = np.load('dataset/data_4d_reshape.npy')   # 20355
label = np.load('dataset/label.npy')

data = torch.FloatTensor(data)
label = torch.FloatTensor(label)

print(data.shape)
print(label.shape)

# 常规数据集划分
# X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=seed)
# print("训练集测试集已划分完成............")
#
# trainData = Data.TensorDataset(X_train, Y_train)
# testData = Data.TensorDataset(X_test, Y_test)
# train_dataloader = DataLoader(trainData, batch_size=batch_size)
# test_dataloader = DataLoader(testData, batch_size=batch_size)
# print("dataloader已完成装载............")

# 五折交叉验证
train_dataloader, test_dataloader = myDataset_5cv(data, label, batch_size, n, seed)
# train_dataloader, test_dataloader = myDataset_10cv(data, label, batch_size, n, seed)
