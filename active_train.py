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

# get GPU or CPU
device = get_device()
args = get_args()
train_data, pool_data, test_data = get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)

num_labels = test_data.label_length()
# print(test_label_length)
out_size = num_labels
num_features = train_data.x.size(1)


def get_new_resnet18():
    active_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                            drop_p=args.m_drop_p, activation=args.m_activation).to(device)
    return active_model


class ActiveLearning:
    def __init__(self, training_data: MyDataset, pooling_data: MyDataset):
        self.dataset = pooling_data
        self.new_dataset = training_data

    def select_instances_mi(self, model, n_instances):
        # Calculate the predictions for all instances in the pool dataset
        with torch.no_grad():
            logits = model(self.dataset.x)
            probs = F.softmax(logits, dim=1)

        # Calculate the mutual information for each instance
        mutual_info_list = []
        for i in range(len(self.dataset)):
            instance_probs = probs[i]
            instance_labels = self.dataset.y[i]
            p_xy = torch.sum(instance_probs * instance_labels)
            p_x = torch.sum(instance_probs)
            p_y = torch.sum(instance_labels)
            mutual_info = p_xy * torch.log(p_xy / (p_x * p_y) + 1e-9)  # Adding a small value to avoid division by zero
            mutual_info_list.append(mutual_info.item())

        # Select the instances with the highest mutual information
        mutual_info_tensor = torch.tensor(mutual_info_list)
        _, selected_indices = torch.topk(mutual_info_tensor, k=n_instances, largest=True)

        # Get the corresponding instances and labels
        new_training_data = self.dataset.x[selected_indices]
        new_training_labels = self.dataset.y[selected_indices]

        # Append the new instances to the new dataset
        self.new_dataset.x = torch.cat([self.new_dataset.x, new_training_data])
        self.new_dataset.y = torch.cat([self.new_dataset.y, new_training_labels])

        # Remove the selected instances from the pool dataset
        indices_to_keep = [i for i in range(len(self.dataset)) if i not in selected_indices]
        self.dataset.x = torch.index_select(self.dataset.x, 0, torch.tensor(indices_to_keep))
        self.dataset.y = torch.index_select(self.dataset.y, 0, torch.tensor(indices_to_keep))

    def select_instances_random(self, n_instances):

        # Get the indices of all instances in the dataset
        all_indices = torch.arange(len(self.dataset))

        # Shuffle the indices
        shuffled_indices = all_indices.tolist()
        random.shuffle(shuffled_indices)

        # Select the first n_instances shuffled indices
        selected_indices = shuffled_indices[:n_instances]

        # Get the corresponding instances and labels
        new_training_data = self.dataset.x[selected_indices]
        new_training_labels = self.dataset.y[selected_indices]

        # Append the new instances to the new dataset
        self.new_dataset.x = torch.cat([self.new_dataset.x, new_training_data])
        self.new_dataset.y = torch.cat([self.new_dataset.y, new_training_labels])

        # Remove the selected instances from the pool dataset
        indices_to_keep = [i for i in range(len(self.dataset)) if i not in selected_indices]
        self.dataset.x = torch.index_select(self.dataset.x, 0, torch.tensor(indices_to_keep).to(torch.int64))
        self.dataset.y = torch.index_select(self.dataset.y, 0, torch.tensor(indices_to_keep).to(torch.int64))

        print(self.new_dataset.__len__())

    def select_instances_entropy(self, model, n_instances):
        # Calculate the entropy of predictions for all instances in the pool dataset
        with torch.no_grad():
            logits = model(self.dataset.x)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs), dim=1)

        # Select the instances with the highest entropy
        _, selected_indices = torch.topk(entropy, n_instances, largest=False)

        # Get the corresponding instances and labels
        new_training_data = self.dataset.x[selected_indices]
        new_training_labels = self.dataset.y[selected_indices]

        # Append the new instances to the new dataset
        self.new_dataset.x = torch.cat([self.new_dataset.x, new_training_data])
        self.new_dataset.y = torch.cat([self.new_dataset.y, new_training_labels])

        # Remove the selected instances from the pool dataset
        self.dataset.x = torch.cat([self.dataset.x[:selected_indices[0]], self.dataset.x[selected_indices[0] + 1:]],
                                   dim=0)
        self.dataset.y = torch.cat([self.dataset.y[:selected_indices[0]], self.dataset.y[selected_indices[0] + 1:]],
                                   dim=0)

        print(self.new_dataset.__len__())

        #  return self.new_dataset, self.dataset

    def train_model(self, model, epochs=args.active_epochs):

        criterion = ml_nn_loss2
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        al_data_loader = DataLoader(self.new_dataset, batch_size=30, shuffle=True)
        for epoch in range(epochs):
            running_loss = 0.0
            for data in al_data_loader:
                inputs, labels = data
                optimizer.zero_grad()
                batch_outputs = model(inputs)
                loss = criterion(labels, batch_outputs, model, device=device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(al_data_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")





