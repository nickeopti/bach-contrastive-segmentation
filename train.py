import os.path
from argparse import ArgumentParser
from typing import Dict, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import arguments
import blocks
import data
import models
import selection


def parse_model_arguments(arg_parser: ArgumentParser) -> Dict[str, arguments.ClassArguments]:
    available_samplers: List[selection.Sampler] = [
        selection.UniformSampler,
        selection.ProbabilisticSampler,
        selection.TopKSampler,
    ]
    available_attention_networks: List[torch.nn.Module] = [
        blocks.AttentionNetwork
    ]
    available_feature_networks: List[torch.nn.Module] = [
        blocks.TypeNetwork
    ]
    options = {
        "sampler": available_samplers,
        "attention_network": available_attention_networks,
        "feature_network": available_feature_networks,
    }
    return {
        key: arguments.add_options(arg_parser, key, values) for key, values in options.items()
    }


def create_trainer(args) -> pl.Trainer:
    logger = pl.loggers.CSVLogger("logs")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(every_n_epochs=1)],
    )
    return trainer


def train(model, trainer, args):
    dataset = data.SamplesDataset(args.dataset, crop_size=500)
    train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    try:
        trainer.fit(model, train_data_loader)
    except RuntimeError as exc:
        if "element 0 of tensors does not require grad" not in str(exc):
            raise exc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="image.tiff")
    parser.add_argument("--model_checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1)

    model_parameters = parse_model_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    trainer = create_trainer(args)

    successfully_started = False
    while not successfully_started:
        model = models.Model(
            counter=models.Counter(args.out_channels, 50, 20),
            inter_channel_loss_scaling_factor=1,
            gamma=args.gamma,
            **{parameter: ca.class_type(**ca.arguments) for parameter, ca in model_parameters.items()}
        )

        if args.model_checkpoint is not None:
            model.load_from_checkpoint(args.model_checkpoint)

        train(model, trainer, args)

        successfully_started = os.path.exists(f"{trainer.logger.log_dir}/metrics.csv")
