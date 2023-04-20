from typing import Dict

import torch
from torch import Tensor, nn
from torchvision import models


class ResNet(nn.Module):
    """
    Torchvision two headed ResNet and ResNext configuration
    """
    def __init__(
            self,
            num_classes: int,
            restype: str = "ResNet18",
            pretrained: bool = False,
            freezed: bool = False,
            ff: bool = False,
        ) -> None:
        super().__init__()
        #Resnet 101
        torchvision_model = models.resnext101_32x8d(pretrained=pretrained)
        self.ff = ff
