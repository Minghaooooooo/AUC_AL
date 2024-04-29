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

    def select_instances_dropout(self, model, n_instances, n_samples=10):
        # List to store predictions for each instance
        all_predictions = []

        # Perform inference with dropout enabled multiple times
        for _ in range(n_samples):
            with torch.no_grad():
                # Enable dropout during inference
                model.train()

                # Forward pass to get predictions
                logits = model(self.dataset.x)
                probs = torch.sigmoid(logits)

            all_predictions.append(probs.unsqueeze(0))  # Store predictions for each sample

        print(len(all_predictions))

        # Concatenate predictions along the sample dimension
        all_predictions = torch.cat(all_predictions, dim=0)

        print(all_predictions.shape)

        # Calculate uncertainty as the variance across samples for each instance
        uncertainty = torch.var(all_predictions, dim=0).sum(dim=1)

        print('uncertainty', uncertainty.shape)

        # Select the instances with the highest uncertainty
        _, selected_indices = torch.topk(uncertainty, n_instances)

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

    def select_instances_entropy(self, model, n_instances):
        # Calculate the entropy of predictions for all instances in the pool dataset
        with torch.no_grad():
            logits = model(self.dataset.x)
            # print("Logits shape:", logits.shape)
            probs = torch.sigmoid(logits)  # Using sigmoid instead of softmax
            # probs = F.softmax(logits, dim=1)

        # # margin is not suitable for multi-lable classification
        # margins, _ = torch.topk(probs, 2, dim=1)
        # margin = margins[:, 0] - margins[:, 1]  # Difference between top two probabilities
        # _, selected_indices = torch.topk(margin, n_instances)

        # # multiple margin
        # # Calculate the number of positive labels in new_training_data
        # avg_num_pos_labels = torch.mean(torch.sum(self.new_dataset.y, dim=1)).item()
        # print(avg_num_pos_labels)
        # # Calculate the top m margins for each instance0
        # margins, _ = torch.topk(probs, int(avg_num_pos_labels), dim=1)
        # margins_diff = margins[:, :-1] - margins[:, 1:]
        # # Calculate the mean margin across labels for each instance
        # xi = int(avg_num_pos_labels/2)
        # # Create a tensor representing the indices of margins_diff
        # indices = torch.arange(margins_diff.size(1), dtype=torch.float).unsqueeze(0)
        # # Compute the power term for each element
        # power_term = xi - indices
        # # Compute margins_diff multiplied by e to the power of (xi - index)
        # margins_diff_exp = margins_diff * torch.exp(power_term)
        # print('margin_diff_exp.shape', margins_diff_exp.shape)
        # sum_margin = torch.sum(margins_diff_exp, dim=1)
        # print('mean_margin', sum_margin.shape)
        # # Select the instances with the highest mean margin
        # _, selected_indices = torch.topk(sum_margin, n_instances, largest=False)
        # # _, selected_indices = torch.topk(margin, n_instances, largest=True)

            # confidences, _ = torch.max(probs, dim=1)
        #     least_confidence = 1 - confidences  # Confidence is inversely related to uncertainty
        # _, selected_indices = torch.topk(least_confidence, n_instances)

        #     # entropy select
        #     entropy = -torch.sum(probs * torch.log(probs), dim=1)
        # _, selected_indices = torch.topk(entropy, n_instances)  # , largest=False)

        # random select
        total_instances = len(self.dataset)
        indices = list(range(total_instances))
        random.shuffle(indices)
        selected_indices = indices[:n_instances]

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

        print(self.new_dataset.__len__())
        print(self.dataset.__len__())

    def train_model(self, model, epochs=args.active_epochs):

        criterion = ml_nn_loss2
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        al_data_loader = DataLoader(self.new_dataset, batch_size=args.active_batch_size, shuffle=True)
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

