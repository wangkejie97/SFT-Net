import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random


# This is toolbox
# There are some function:
# 1.label_2class
# 2.myEvaluate
# 3.myDataset_5cv
# 4.myDataset_5cv_validation


# description: Convert list of raw or predicted labels to two-class, default threshold is 0.35 in SEED-VIG and paper
# input: List[int]
# output: List[int]
def label_2class(a):
    label_2classes = []
    # a = torch.squeeze(a)
    for i in range(0, len(a)):
        if a[i] < .35:
            label_2classes.append(0)
        else:
            label_2classes.append(1)
    return label_2classes


# description: Evaluate the performance of the saved model
# input: List[int], List[int] (true label and predicted label)
# output: Confusion matrix, Accuracy, Report, Precision, Recall, F1-score, Kappa
def myEvaluate(true, pred):
    if len(true) != len(pred):
        return False
    else:
        conf = confusion_matrix(true, pred)
        acc = accuracy_score(true, pred)
        report = classification_report(true, pred, target_names=['Awake', 'Fatigue'])
        pre = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = 2*(pre * recall)/(pre + recall)
        kappa = cohen_kappa_score(true, pred)
    return conf, acc, report, pre, recall, f1, kappa


# description: 5-fold cross validation
# input: data, label, batch_size n:return fold, seed:random seed
# output: train_dataloader, test_dataloader
def myDataset_5cv(data, label, batch_size, n, seed):
    lenth = data.shape[0]
    a = lenth//5
    index = []
    for i in range(lenth):
        index.append(i)
    random.seed(seed)
    random.shuffle(index)
    index_1 = index[0:a]
    index_2 = index[a:2*a]
    index_3 = index[2*a:3*a]
    index_4 = index[3*a:4*a]
    index_5 = index[4*a:-1]
    if n == 0:
        train_index = index_2 + index_3 + index_4 + index_5
        test_index = index_1
    if n == 1:
        train_index = index_1 + index_3 + index_4 + index_5
        test_index = index_2
    if n == 2:
        train_index = index_1 + index_2 + index_4 + index_5
        test_index = index_3
    if n == 3:
        train_index = index_1 + index_2 + index_3 + index_5
        test_index = index_4
    if n == 4:
        train_index = index_1 + index_2 + index_3 + index_4
        test_index = index_5

    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    train_dataloader = DataLoader(Data.TensorDataset(X_train, y_train), batch_size=batch_size)
    test_dataloader = DataLoader(Data.TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_dataloader, test_dataloader


# description: 5-fold cross validation only use test data for evaluating models
# input: data, label, n:return fold, seed:random seed
# output: X_test, y_test
def myDataset_5cv_validation(data, label, n, seed):
    lenth = data.shape[0]
    a = lenth//5
    index = []
    for i in range(lenth):
        index.append(i)
    random.seed(seed)
    random.shuffle(index)
    index_1 = index[0:a]
    index_2 = index[a:2*a]
    index_3 = index[2*a:3*a]
    index_4 = index[3*a:4*a]
    index_5 = index[4*a:-1]
    if n == 0:
        train_index = index_2 + index_3 + index_4 + index_5
        test_index = index_1
    if n == 1:
        train_index = index_1 + index_3 + index_4 + index_5
        test_index = index_2
    if n == 2:
        train_index = index_1 + index_2 + index_4 + index_5
        test_index = index_3
    if n == 3:
        train_index = index_1 + index_2 + index_3 + index_5
        test_index = index_4
    if n == 4:
        train_index = index_1 + index_2 + index_3 + index_4
        test_index = index_5

    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    return X_test, y_test


if __name__ == '__main__':
    # debug
    print("This is 4D-A-DSC-LSTM toolbox!")