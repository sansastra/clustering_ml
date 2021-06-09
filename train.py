import torch
import torch.nn as nn
import numpy as np
import copy
import time
from torch.autograd import Variable


def statistics(epoch, phase, losses, accuracy, epoch_loss, epoch_acc):
    losses["loss"].append(epoch_loss)
    losses["phase"].append(phase)
    losses["epoch"].append(epoch)

    accuracy["accuracy"].append(epoch_acc)
    accuracy["epoch"].append(epoch)
    accuracy["phase"].append(phase)


def train_model(model, x_train, x_test, y_train, y_test, num_classes, criterion, optimizer, scheduler=None,
                num_epochs=25, device=None, uncertainty=False):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}


    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                data_x = Variable(torch.from_numpy(np.array(x_train)).float())
                data_y = Variable(torch.from_numpy(np.array(y_train)).float())
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                data_x = Variable(torch.from_numpy(np.array(x_test)).float())
                data_y = Variable(torch.from_numpy(np.array(y_test)).float())
                model.eval()   # Set model to evaluate mode

            if num_classes == 1:
                data_y = data_y.reshape(shape=(len(data_y), 1))
            running_loss = 0.0
            nr_success = 0
            nr_fail = 0
            acc = 0

            # Iterate over data.
            batch_size = 8
            total_run = len(data_x)//batch_size - 1
            for i in range(total_run):
                inputs = data_x[i*batch_size:(i+1)*batch_size,:]
                labels = data_y[i*batch_size:(i+1)*batch_size,:]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    match = torch.reshape(torch.eq(
                        preds, torch.argmax(labels, dim=1)).float(), (-1, 1))
                    acc += torch.mean(match)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels[:,1].data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / total_run
            epoch_acc = acc / total_run

            statistics(epoch, phase, losses, accuracy,  epoch_loss, epoch_acc)

            print("{} loss: {:.4f} acc: {:.4f}".format(
                phase.capitalize(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
