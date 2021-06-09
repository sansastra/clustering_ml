import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt


def plot_train_metrics(metrics, K):
    num_epochs = 2*(metrics[0]["epoch"][-1] + 1) # for training and validation
    train_acc1 = metrics[1]["accuracy"][0:num_epochs:2]
    test_acc1 = metrics[1]["accuracy"][1:num_epochs:2]

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches([6, 4])

    axs[0].plot(train_acc1, c='blue', marker='*')
    axs[0].plot(test_acc1, c='green', marker='+')
    axs[0].set_title('Train Data')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Test'])
    axs[0].set_ybound(0, 1)
    fig.tight_layout()
    plt.savefig("./results/cluster_all.png")
    plt.savefig("./results/cluster_all.pdf")
    plt.show()


def test_data(model, x_test, y_test):

    num_classes = y_test.shape[1]
    data_x = Variable(torch.from_numpy(np.array(x_test)).float())
    data_y = Variable(torch.from_numpy(np.array(y_test)).float())
    print("Testing...")
    output = model(data_x)
    _, preds = torch.max(output, 1)
    match = torch.reshape(torch.eq(
        preds, torch.argmax(data_y, dim=1)).float(), (-1, 1))
    acc = torch.mean(match)
    print("test accuracy is ", acc)
    # prob = F.softmax(output, dim=1)
    # output = output.flatten()
    # prob = prob.flatten()
    # preds = preds.flatten()
    # print("Predict:", preds[0])
    # print("Probs:", prob)



