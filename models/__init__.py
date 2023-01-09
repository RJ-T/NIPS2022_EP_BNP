from .resnet import resnet18
import torch.nn as nn


def get_model(model, num_classes, norm_layer=nn.BatchNorm2d):
    if model == 'resnet18':
        return resnet18(num_classes, norm_layer=norm_layer)
    else:
        raise
