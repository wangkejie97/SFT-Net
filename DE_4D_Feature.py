import torch
import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':
    data = np.load("./processedData/data_3d.npy")
    print("data.shape: ", data.shape)  # [325680, 17, 5] -> [20355, 16, 6, 9, 5]

    img_rows, img_cols, num_chan = 6, 9, 5
    data_4d = np.zeros((325680, img_rows, img_cols, num_chan))  # [samples, height, width, channels]->[325680, 6, 9, 5]
    print("data_4d.shape :", data_4d.shape)

    # Sequence of seventeen electrode channels
    channels = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']

    # 'FT7'(channel1) :
    data_4d[:, 0, 0, :] = data[:, 0, :]
    # 'FT8'(channel2) :
    data_4d[:, 0, 8, :] = data[:, 1, :]
    # 'T7' (channel3) :
    data_4d[:, 1, 0, :] = data[:, 2, :]
    # 'T8' (channel4) :
    data_4d[:, 1, 8, :] = data[:, 3, :]
    # 'TP7'(channel5) :
    data_4d[:, 2, 0, :] = data[:, 4, :]
    # 'TP8'(channel6) :
    data_4d[:, 2, 8, :] = data[:, 5, :]
    # 'CP1'(channel7) :
    data_4d[:, 2, 3, :] = data[:, 6, :]
    # 'CP2'(channel8) :
    data_4d[:, 2, 5, :] = data[:, 7, :]
    # 'P1' (channel9) :
    data_4d[:, 3, 3, :] = data[:, 8, :]
    # 'PZ' (channel10):
    data_4d[:, 3, 4, :] = data[:, 9, :]
    # 'P2' (channel11):
    data_4d[:, 3, 5, :] = data[:, 10, :]
    # 'PO3'(channel12):
    data_4d[:, 4, 3, :] = data[:, 11, :]
    # 'POZ'(channel13):
    data_4d[:, 4, 4, :] = data[:, 12, :]
    # 'PO4'(channel14):
    data_4d[:, 4, 5, :] = data[:, 13, :]
    # 'O1' (channel15):
    data_4d[:, 5, 3, :] = data[:, 14, :]
    # 'OZ' (channel16):
    data_4d[:, 5, 4, :] = data[:, 15, :]
    # 'O2' (channel17):
    data_4d[:, 5, 5, :] = data[:, 16, :]

    np.save('./processedData/data_4d.npy', data_4d)