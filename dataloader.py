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
data = np.load('./processedData/data_4d.npy')   # 20355
label = np.load('./processedData/label.npy')

data = torch.FloatTensor(data)
label = torch.FloatTensor(label)

print(data.shape)
print(label.shape)

# 五折交叉验证
train_dataloader, test_dataloader = myDataset_5cv(data, label, batch_size, n, seed)
