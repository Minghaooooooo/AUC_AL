import copy
import csv
import time

from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from config import get_args
from loss import ml_nn_loss2
from util import *
from architecture import *
from data import get_data, MyDataset
import torch.nn.functional as F
from sampling_function import *

# get GPU or CPU
args = get_args()
train_data, pool_data, test_data, train_origin, pool_origin = get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)

num_labels = test_data.label_length()
# print(test_label_length)
out_size = num_labels
num_features = train_data.x.size(1)


def get_new_resnet18():
    active_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                            drop_p=args.m_drop_p, activation=args.m_activation)
    return active_model



class ActiveLearning:
    def __init__(self, train_origin, pool_origin, training_data: MyDataset, pooling_data: MyDataset, device_al_train):
        self.pool_dataset = pooling_data
        self.train_dataset = training_data
        self.device_al_train = device_al_train
        self.train_origin = train_origin
        self.pool_origin = pool_origin

    def select_instances(self, model, n_instances):
        # Calculate the entropy of predictions for all instances in the pool dataset
        with torch.no_grad():
            model = model.cpu()
            logits = model(self.pool_dataset.x.cpu())
            # print("Logits shape:", logits.shape)
            probs = torch.sigmoid(logits)  # Using sigmoid instead of softmax
            # probs = F.softmax(logits, dim=1)

            # confidences, _ = torch.max(probs, dim=1)
        #     least_confidence = 1 - confidences  # Confidence is inversely related to uncertainty
        # _, selected_indices = torch.topk(least_confidence, n_instances)

        #     # entropy select
        #     entropy = -torch.sum(probs * torch.log(probs), dim=1)
        # _, selected_indices = torch.topk(entropy, n_instances)  # , largest=False)

        # selected_indices = multiple_margin(self, probs, n_instances)
        # selected_indices = test11(self, probs, n_instances)
        # selected_indices = random_sampling(self, probs, n_instances)
        # selected_indices = CVIRS_sampling(self,  probs, n_instances)
        selected_indices = test15(self, probs, n_instances)

        # Get the corresponding instances and labels
        new_training_data = self.pool_dataset.x[selected_indices]
        new_training_labels = self.pool_dataset.y[selected_indices]
        new_training_origin_data = self.pool_origin[selected_indices]

        # Append the new instances to the new dataset
        self.train_dataset.x = torch.cat([self.train_dataset.x, new_training_data])
        self.train_dataset.y = torch.cat([self.train_dataset.y, new_training_labels])
        self.train_origin = torch.cat([self.train_origin, new_training_origin_data])

        # Remove the selected instances from the pool dataset
        indices_to_keep = [i for i in range(len(self.pool_dataset)) if i not in selected_indices]
        self.pool_dataset.x = torch.index_select(self.pool_dataset.x, 0, torch.tensor(indices_to_keep))
        self.pool_dataset.y = torch.index_select(self.pool_dataset.y, 0, torch.tensor(indices_to_keep))
        self.pool_origin = torch.index_select(self.pool_origin, 0, torch.tensor(indices_to_keep))

        print('There are ', self.train_dataset.__len__(), ' samples in training data')
        print('We have ', self.pool_dataset.__len__(), ' samples can choose' )


    def train_model(self, model, epochs=args.active_epochs):
        model = model.to(self.device_al_train)
        criterion = ml_nn_loss2
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        al_data_loader = DataLoader(self.train_dataset, batch_size=args.active_batch_size, shuffle=True)
        for epoch in range(epochs):
            running_loss = 0.0
            for (inputs, labels) in al_data_loader:
                inputs = inputs.to(self.device_al_train)
                labels = labels.to(self.device_al_train)
                optimizer.zero_grad()
                batch_outputs = model(inputs)
                loss = criterion(labels, batch_outputs, model, device=self.device_al_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(al_data_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

