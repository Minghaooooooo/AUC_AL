import os

import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import csv
from config import get_args


def add_res(model_opt, x_test, y_test, n=None,
            xtest=None, ytest=None, device=None):
    if xtest is None:
        xtest = x_test.clone().detach()
        ytest = y_test.clone().detach().requires_grad_(True)

    output = model_opt(xtest.float().to(device))
    ground_truth = ytest
    mse = mean_squared_error(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy())

    # Convert output to binary predictions (for binary classification)
    predictions = torch.round(output).detach().cpu().numpy()

    f1 = f1_score(ground_truth.detach().cpu().numpy(), predictions, average=None)

    macro_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy(), average='macro')
    micro_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy(), average='micro')
    file_path = './result/data.csv'
    for i, j in zip(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy()):
        a = []
        for element_gt, element_output in zip(i, j):
            if element_gt == 1:
                a += [element_gt]
                a += [element_output]
        with open(file_path, 'a', newline='') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(a)

    return mse, micro_auc, macro_auc
