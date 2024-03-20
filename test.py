import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error


def print_res(model_opt, x_test, y_test, n=None,
              xtest=None, ytest=None, device=None):
    if xtest is None:
        xtest = x_test.clone().detach()
        ytest = y_test.clone().detach().requires_grad_(True)

    output = model_opt(xtest.float().to(device))
    ground_truth = ytest
    mse = mean_squared_error(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy())

    datapoint = [n, mse]
    return datapoint
