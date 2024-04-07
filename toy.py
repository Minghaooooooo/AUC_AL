from loss import *

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
from architecture import *


def check_cv_and_preprocess_data(model):
    # Define a new list to reserve CV models
    cv_models = []

    # Add ViTModel to the CV_models list
    vit_model = ViTModel(in_size=None, hidden_size=224, out_size=224, embed=224,
                 drop_p=0.5, activation=None)  # Instantiate ViTModel with appropriate arguments
    cv_models.append(type(vit_model))
    res18_model = ResNet18(in_size=224, hidden_size=224, out_size=224, embed=224,
                          drop_p=0.5, activation=None)  # Instantiate ViTModel with appropriate arguments
    cv_models.append(type(res18_model))
    res50_model = ResNet50(in_size=224, hidden_size=224, out_size=224, embed=224,
                          drop_p=0.5, activation=None)  # Instantiate ViTModel with appropriate arguments
    cv_models.append(type(res50_model))
    # Add other CV models to the list if needed

    def is_cv_model(model_check):
        """
        Check if the given model belongs to the CV models list.

        Args:
            model_check (nn.Module): The model to check.

        Returns:
            bool: True if the model belongs to the CV models list, False otherwise.
        """
        return type(model_check) in cv_models

    if is_cv_model(model):
        # Handle CV model training differently
        print("yes")
    else:
        # Handle non-CV model training
        print("no")


# vit_model1 = ViTModel(in_size=None, hidden_size=224, out_size=224, embed=224,
#                  drop_p=0.5, activation=None)  # Instantiate ViTModel with appropriate arguments

vit_model1 = ResNet50(in_size=224, hidden_size=224, out_size=224, embed=224,
                 drop_p=0.5, activation=None)  # Instantiate ViTModel with appropriate arguments

check_cv_and_preprocess_data(vit_model1)
