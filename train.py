from myNet import *
from toolbox import *
from visdom import Visdom
from dataloader import *
import warnings
warnings.filterwarnings("ignore")

# Pay attention!
# Before train begin, start up the visdom to Real-time monitoring accuracy


# save highest accuracy model.pth
acc_low = 0.8
acc_list = []
# model
myModel = My_4D_A_DSC_LSTM()
myModel = myModel.to(device)
# loss function
loss_fn = torch.nn.MSELoss()
# optimizer
learningRate = 2e-3
optimizer = torch.optim.AdamW(myModel.parameters(), lr=learningRate, weight_decay=0.02) # AdamW

epoch = 200
# Record total step and loss
total_train_step = 0
total_test_step = 0
total_train_loss = 0
total_test_loss = 0
# Visdom
viz = Visdom()
train_loss_viz = 0
test_loss_viz = 0
acc_viz = 0
total_train_viz = 0
total_test_viz = 0
# Visdom init
viz.line([[1, 1]], [0], win='loss', opts=dict(title='loss', legend=['train', 'test']))
viz.line([[0, 0]], [0], win='acc', opts=dict(title='acc', legend=['train', 'test']))


for i in range(epoch):
    print("--------------The {}th epoch of training starts------------".format(i + 1))
    total_train_loss = 0
    total_train_acc = 0
    ### 1.TRAIN ###
    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        # ignore attention output when training
        outputs, _sa, _fa = myModel(x)
        train_loss = loss_fn(outputs, y)
        # Calculate accuracy by toolbox.label_2class
        label_train = label_2class(y)
        label_train_pred = label_2class(outputs)
        total_train_acc += accuracy_score(label_train, label_train_pred)
        # Gradient update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Calculate loss
        total_train_loss = total_train_loss + train_loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("train step：{}，train average loss：{:.6f}".format(total_train_step, total_train_loss/total_train_step))
    total_train_viz += 1
    # in this epoch, train accuracy
    print("train average accuracy: {:.4f}%".format(100.0 * (total_train_acc/len(train_dataloader))))

    ### 2.TEST ###
    total_test_acc = 0
    total_test_loss = 0
    r2 = 0
    attentionGraph = torch.Tensor()
    with torch.no_grad():
        for data in test_dataloader:
            testx, testy = data
            testx = testx.to(device)
            testy = testy.to(device)
            outputs, spaAtten, freqAtten = myModel(testx)
            label = label_2class(testy)
            label_pred = label_2class(outputs)
            test_loss = loss_fn(outputs, testy)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_step = total_test_step + 1
            # use toolbox.myEvaluate to calculate
            conf, acc, report, pre, recall, f1, kappa = myEvaluate(label, label_pred)
            # total accuracy
            total_test_acc += acc

    total_test_viz += 1
    print("test average loss:{:.6f}".format(total_test_loss / total_test_step))
    # in this epoch test accuracy
    print("test average accuracy:{:.4f}%".format(100.0 * (total_test_acc / len(test_dataloader))))

    viz.line([[total_train_loss/len(train_dataloader), total_test_loss/len(test_dataloader)]], [i+1], win='loss', update="append")
    viz.line([[total_train_acc/len(train_dataloader), total_test_acc/len(test_dataloader)]], [i+1], win='acc', update="append")

    #  If test accuracy more than the acc_low, save the model.pth
    if (total_test_acc/len(test_dataloader)) > acc_low:
        acc_low = (total_test_acc/len(test_dataloader))
        torch.save(myModel.state_dict(), './pth/model_fold_%d.pth' % n)
print("This is ", n, " fold, highest accuracy is: ", acc_low)