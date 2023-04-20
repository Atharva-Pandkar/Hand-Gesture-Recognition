import argparse
import random
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import omegaconf, DictConfig
from torchvision.models import ResNet


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Gesture classifier...")

    parser.add_argument(
        "-c", "--command", required=True, type=str, help="Training or test pipeline", choices=("train", "test")
    )

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


def _initialize_model(conf: DictConfig):
    random_seed = conf.random_state
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(conf.dataset.targets,conf.model.name,conf.device)
    num_classes = len(conf.dataset.targets)
    conf.num_classes = {"gesture": num_classes, "leading_hand": 2}

    model = ResNet(num_classes=num_classes, restype="ResNet152", pretrained=False, freezed=False, ff=False)

    return model


args = parse_arguments()
path_to_config = args.path_to_config
if args.command == 'train':
    print("Inside Run_train function")
    conf = omegaconf.OmegaConf.load(path_to_config)
    print("Going to initialize model")
    model = _initialize_model(conf)
    print("Model Initialised")
    print("Going inside the Gesture Dataset")
    train_dataset = GestureDataset(is_train=True, conf=conf, transform=get_transform())

    test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform())

    logging.info(f"Current device: {conf.device}")
    TrainClassifier.train(model, conf, train_dataset, test_dataset)