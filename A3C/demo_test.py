import torch
from torch import nn
from torchvision import models
from torch import optim

resnet_model = models.resnet18(pretrained=False)
for name, para in resnet_model.named_parameters():
    print(name)

optim.SGD()

# nn.Module
