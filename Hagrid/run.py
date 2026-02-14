import argparse
import logging
from typing import Optional, Tuple

import torch.optim
import torch.utils
import omegaconf.omegaconf
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

import Hagrid.utils
from Hagrid.dataset import GestureDataset
from Hagrid.preprocess import get_transform
from Hagrid.train import TrainClassifier
from Hagrid.utils import build_model, collate_fn, set_random_state

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


def _initialize_model(conf: DictConfig):
    set_random_state(conf.random_state)
    logging.info(f"{conf.dataset.targets}, {conf.model.name}, {conf.device}")
    num_classes = len(conf.dataset.targets)
    conf.num_classes = {"gesture": num_classes, "leading_hand": 2}

    model = Hagrid.utils.build_model_train(
        model_name=conf.model.name,
        num_classes=num_classes,
        checkpoint=conf.model.get("checkpoint", None),
        device=conf.device,

    )

    return model


def _run_test(path_to_config: str):
    """
    Run training pipeline

    Parameters
    ----------
    path_to_config : str
        Path to config
    """
    conf = omegaconf.OmegaConf.load(path_to_config)
    model = _initialize_model(conf)

    experimnt_pth = f"experiments/{conf.experiment_name}"
    writer = SummaryWriter(log_dir=f"{experimnt_pth}/logs")
    writer.add_text("model/name", conf.model.name)

    test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform(), is_test=True)

    logging.info(f"Current device: {conf.device}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.train_params.test_batch_size,
        num_workers=conf.train_params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=conf.train_params.prefetch_factor,
    )

    TrainClassifier.eval(model, conf, 0, test_dataloader, writer, "test")


def _run_train(path_to_config: str) -> None:
    """
    Run training pipeline

    Parameters
    ----------
    path_to_config : str
        Path to config
    """
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


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Gesture classifier...")

    parser.add_argument(
        "-c", "--command", required=True, type=str, help="Training or test pipeline", choices=("train", "test")
    )

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    if args.command == "train":
        print("Performing Train operations")
        _run_train(args.path_to_config)
    elif args.command == "test":
        _run_test(args.path_to_config)
