

import torch

print("PyTorch version:", torch.__version__)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
print(get_device())

