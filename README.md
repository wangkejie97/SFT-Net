# SFT-Net

#### Introduction

1. The title of the paper is: "SFT-Net:Spatial–Frequency–Temporal Network based on attention Mechanism for Detecting Driver Fatigue From EEG Signals"

2. The original address of the paper is:

3. The GITHUB address of this project is: https://github.com/wangkejie97/SFT-Net

   

#### Requirements

​	In this folder path, use the CMD command to the terminal, (if it is a virtual environment, please switch first), and then execute the following code to install the dependency package.

```
pip install -r requirements.txt
```

It contains the following five packages.

- numpy==1.19.5
- scikit_learn==1.0.2
- scipy==1.5.4
- torch==1.9.0
- visdom==0.1.8.9



#### File and folder contents

- ***DE_3D_Feature.py*** : Convert raw EEG data of 23 subjects to 3D features.
- ***DE_4D_Feature.py*** : Convert 3D features into 4D features according to the 2D topographic map (refer to the paper).
- ***dataloader*** : Divide the four-dimensional features and dataset labels into training set (4/5) and test set (1/5) according to the custom five-fold cross-validation.
- ***train*** : training and testing, the training curve can be displayed in real time on the web page through visdom.
- ***myNet*** : the defined SFT-Net model.
- ***"./processedData/"*** : used to store the converted 3D features and 4D features.
- ***"./pth/"*** : used to store the model with the highest accuracy in the nth fold training.



#### Quick start

1. Open "SFT-Net/DE_3D_Feature", at line 92, replace with the actual data set path in your computer, then run the py file, it will be in "4D-A-DSC- The "data_3d.npy" file is generated under LSTM/processedData".

2. Open "SFT-Net/DE_4D_Feature" and run it directly. After completion, the "data_4d.npy" file will be generated under "SFT-Net/processedData".

3. Open "SFT-Net/dataloader", you can adjust the number of folds in the five-fold cross-validation for verification, set batch_size, or set a random number seed.

4. Open "SFT-Net/train", before starting the training, please open the cmd command line, (if using a virtual environment, please switch first), enter

   ```
   python -m visdom.server
   ```

Then, open the website in the prompt for real-time visualization. You can adjust the learning rate or Epoch yourself.



##### Others

- Attention visualization can be obtained by spaAtten, freqAtten output by the model network.
- Requires visdom to be turned on while training.