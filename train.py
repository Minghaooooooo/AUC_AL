import copy
import time

from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from config import get_args
from util import *
from model import *

from data import get_data


def train_sep_bm(model,
                 dataloaders,
                 criterion,
                 optimizer,
                 scheduler=None,
                 num_epochs=None,
                 device_train=None,
                 num_l=None,
                 fname=None
                 ):
    since = time.time()

    if not device_train:
        device_train = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            for phase in ["train"]:  # , "val"]:
                if phase == "train":
                    print("Training...")
                    model.train()  # Set model to training mode
                else:
                    print("Validating...")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                for i, (inputs, labels) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device_train)
                    labels = labels.to(device_train)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        y = labels.to(device_train)
                        # print('inputs',inputs)
                        # print('labels',labels)
                        outputs = model(inputs)
                        # print('outputs',outputs)
                        loss = criterion(y.float(), outputs, device=device_train)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                # preds = predict(model,inputs)
                # match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                # acc = torch.mean(match)
                # print(acc)

                if scheduler is not None:
                    if phase == "train":
                        scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('epoch loss', epoch_loss)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f'Parameter: {name}, Gradient: {param.grad.mean()}')
                    else:
                        print(f'Parameter: {name} has no gradient')

                loss_list.append(epoch_loss)
            print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # metrics = (losses, accuracy)
    return model, loss_list