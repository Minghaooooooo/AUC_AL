import torch
from torch import nn
import torch.nn.functional as F
from util import get_device


# def pairwise_ranking_loss(y_true, y_pred):
#     # 根据模型输出排序
#     sorted_indices = torch.argsort(y_pred, dim=0, descending=True)
#     sorted_labels = y_true[sorted_indices]
#
#     # 找到正样本和负样本的索引
#     positive_indices = torch.where(sorted_labels == 1)[0]
#     negative_indices = torch.where(sorted_labels == 0)[0]
#
#     # 确保正样本和负样本索引都不为空
#     if len(positive_indices) == 0 or len(negative_indices) == 0:
#         return torch.tensor(0.0)  # 如果没有找到正样本或负样本，返回零损失
#
#     # 计算 pairwise ranking loss
#     y_pred_positive = y_pred[positive_indices][:, None]
#     y_pred_negative = y_pred[negative_indices][None, :]
#     loss = torch.sum(torch.maximum(torch.tensor(0.0), 1.0 - y_pred_negative + y_pred_positive))
#
#     return loss


def pairwise_ranking_loss(y_true, y_pred):
    """
    Compute pairwise ranking loss.

    Parameters:
        y_true (tensor): True labels (binary, 0 or 1).
        y_pred (tensor): Predicted scores.

    Returns:
        loss (tensor): Pairwise ranking loss.
    """
    # Compute differences between pairs of predicted scores
    pairwise_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)

    # Create mask to filter out same pairs
    mask = torch.eye(y_true.size(0), dtype=torch.bool)

    # Compute pairwise ranking loss
    loss = torch.sum(torch.relu(1 - pairwise_diff[mask])) / torch.sum(mask.float())

    return loss


def ml_nn_loss(y, outputs, model, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    # non_mse = nn.MSELoss()
    # Multilabelsoftmarginloss:
    non_mse = nn.MultiLabelSoftMarginLoss()
    # Non Multilabel Loss:
    # non_mse = nn.BCELoss()  # Binary Cross-Entropy Loss

    loss = non_mse(outputs, y)
    # loss = pairwise_ranking_loss(y, outputs)

    # y_pairwise = y.view(-1)
    # outputs_pairwise = outputs.view(-1)
    # print("outputs", outputs_pairwise.shape)
    # print("y", y_pairwise.shape)
    # loss = pairwise_ranking_loss(y_pairwise, outputs_pairwise)  # Pairwise ranking loss
    return loss


def ml_nn_loss1(y, outputs, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    # non_mse = nn.MultiLabelSoftMarginLoss()
    alpha = 0.1  # You can adjust the weight of the surrogate loss
    non_mse = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    loss = non_mse(outputs, y)
    return loss


def ml_nn_loss_regularization(y, outputs, model, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    # Define the original loss function
    non_mse = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    # Calculate the original loss
    loss = non_mse(outputs, y)
    # Add L2 regularization term
    l2_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2
    weight_decay = 0.01
    loss += 0.5 * weight_decay * l2_regularization
    return loss


def ml_nn_loss_surrogate_auc(y, outputs, model, device=None):
    if not device:
        device = get_device()

    y = y.to(device)

    # Calculate the Hinge loss
    loss = F.hinge_embedding_loss(outputs, y.float(), margin=1.0)

    # Add L2 regularization term
    l2_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2

    weight_decay = 0.01
    loss += 0.5 * weight_decay * l2_regularization

    return loss
