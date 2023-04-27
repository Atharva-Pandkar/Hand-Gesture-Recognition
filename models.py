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
        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False


        self.backbone = nn.Sequential(
            torchvision_model.conv1,
            torchvision_model.bn1,
            torchvision_model.relu,
            torchvision_model.maxpool,
            torchvision_model.layer1,
            torchvision_model.layer2,
            torchvision_model.layer3,
            torchvision_model.layer4,
            torchvision_model.avgpool,
        )

        num_features = torchvision_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
        if not self.ff:
            self.leading_hand = nn.Sequential(
                nn.Linear(num_features, 2),
            )

    def forward(self, img: Tensor) -> Dict:
        x = self.backbone(img)
        x = torch.flatten(x, 1)
        gesture = self.classifier(x)

        if self.ff:
            return {"gesture": gesture}
        else:
            leading_hand = self.leading_hand(x)
            return {"gesture": gesture, "leading_hand": leading_hand}