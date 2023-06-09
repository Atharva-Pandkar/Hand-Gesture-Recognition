import logging
import os
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Hagrid.Models.fasterrcnn import FasterRCNN_Mobilenet_large
from Hagrid.Models.model import TorchVisionModel
from Hagrid.Models.resnet import ResNet


def add_metrics_to_tensorboard(writer: SummaryWriter, metrics: Dict, epoch: int, mode: str, target: str) -> None:
    """
    Add metrics to Tensorboard logs

    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard log writer
    metrics : Dict
        Metrics value
    epoch : int
        Number of epoch
    mode : str
        Mode valid or train
    target : str
        Target name: gesture or leading_hand
    """
    print("Inside the sdmasidb")
    logging.info(f"{mode}: metrics for {target}")
    logging.info(metrics)
    for key, value in metrics.items():
        print(f"{key}_{target}/{mode}",value,epoch,"ASDasdasdasdasdasdasdasdasd")
        writer.add_scalar(f"{key}_{target}/{mode}", value, epoch)


def add_params_to_tensorboard(writer: SummaryWriter, params: Dict, epoch: int, obj: str, not_logging: Set) -> None:
    """
    Add optimizer params to Tensorboard logs

    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard log writer
    params : Dict
        Optimizer params for logging
    epoch : int
        Number of epoch
    obj : str
        Optimizer or learning scheduler for params logging
    not_logging : List
        Parameters that should not be logged
    """
    for param, value in params.items():
        #print(f"{obj}/{param}", value, epoch)
        if param not in not_logging:
            #writer.add_scalar(f"{obj}/{param}", value, epoch)
            print(value)


def set_random_state(random_seed: int) -> None:
    """
    Set random seed for torch, numpy, random

    Parameters
    ----------
    random_seed: int
        Random seed from config
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_checkpoint(
    output_dir: str, config_dict: Dict, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, name: str
) -> None:
    """
    Save checkpoint dictionary

    Parameters
    ----------
    output_dir : str
        Path to directory model checkpoint
    config_dict : Dict
        Config dictionary
    model : nn.Module
        Model for checkpoint save
    optimizer : torch.optim.Optimizer
        Optimizer
    epoch : int
        Epoch number
    name : str
        Model name
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir), exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"{name}")

    checkpoint_dict = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config_dict,
    }
    torch.save(checkpoint_dict, checkpoint_path)


def build_model_train(
    model_name: str,
    num_classes: int,
    device: str,
    checkpoint: str = None,
    pretrained: bool = False,
    freezed: bool = False,
    ff: bool = False,
) -> nn.Module:
    """
    Build modela and load checkpoint

    Parameters
    ----------
    model_name : str
        Model name e.g. ResNet18, MobileNetV3_small, Vitb32
    num_classes : int
        Num classes for each task
    checkpoint : str
        Path to model checkpoint
    device : str
        Cpu or CUDA device
    pretrained : false
        Use pretrained model
    freezed : false
        Freeze model layers
    """
    models = {
        "ResNet18": ResNet(num_classes=num_classes, restype="ResNet18", pretrained=pretrained, freezed=freezed, ff=ff),
        "ResNext50": ResNet(
            num_classes=num_classes, restype="ResNext50", pretrained=pretrained, freezed=freezed, ff=ff
        ),
        "ResNext101": ResNet(
            num_classes=num_classes, restype="ResNext101", pretrained=pretrained, freezed=freezed, ff=ff
        ),
        "ResNet152": ResNet(
            num_classes=num_classes, restype="ResNet152", pretrained=pretrained, freezed=freezed, ff=ff
        )
    }

    model = models[model_name]

    if checkpoint is not None:
        checkpoint = os.path.expanduser(checkpoint)
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=torch.device(device))["state_dict"]
            model.load_state_dict(checkpoint)

    model.to(device)
    return model


def collate_fn(batch: List) -> Tuple:
    """
    Collate func for dataloader

    Parameters
    ----------
    batch : List
        Batch of data
    """
    return tuple(zip(*batch))

def build_model(
    model_name: str,
    num_classes: int,
    device: str,
    checkpoint: str = None,
    pretrained: bool = False,
) -> TorchVisionModel:
    """
    Build modela and load checkpoint

    Parameters
    ----------
    model_name : str
        Model name e.g. ResNet18, MobileNetV3_small, Vitb32
    num_classes : int
        Num classes for each task
    checkpoint : str
        Path to model checkpoint
    device : str
        Cpu or CUDA device
    pretrained : false
        Use pretrained model
    freezed : false
        Freeze model layers
    """
    models = {
       "FasterRCNN_mobilenet_large": FasterRCNN_Mobilenet_large(pretrained=pretrained, num_classes=num_classes),
    }

    model = models[model_name]

    if checkpoint is not None:
        checkpoint = os.path.expanduser(checkpoint)
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=torch.device(device))
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
    return model


def collate_fn(batch: List) -> Tuple:
    """
    Collate func for dataloader

    Parameters
    ----------
    batch : List
        Batch of data
    """
    return tuple(zip(*batch))
