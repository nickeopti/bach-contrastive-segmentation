import os.path
from argparse import ArgumentParser
from typing import Dict, List, Optional

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
        selection.ProbabilisticSentinelSampler,
        selection.TopKSampler,
        selection.EntropySampler,
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
        callbacks=[ModelCheckpoint(
            monitor="best_val_mdc",
            save_top_k=5,
            mode="max",
        )],
    )
    return trainer


def train(model, trainer: pl.Trainer, dataset_info: arguments.ClassArguments, validation_dataset_info: Optional[arguments.ClassArguments], args):
    dataset = dataset_info.class_type(**dataset_info.arguments)
    train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if validation_dataset_info:
        validation_dataset = validation_dataset_info.class_type(**validation_dataset_info.arguments)
        validation_data_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=args.num_workers)

    try:
        trainer.fit(model, train_data_loader, val_dataloaders=validation_data_loader if validation_dataset_info else None)
    except RuntimeError as exc:
        if "element 0 of tensors does not require grad" not in str(exc):
            raise exc


if __name__ == "__main__":
    parser = ArgumentParser()
    dataset_info = arguments.add_options(parser, "dataset", (data.SamplesDataset, data.MultiSamplesDataset, data.MoNuSegDataset, data.MoNuSegWSIDataset, data.GlaSDataset, data.CoNSePDataset, data.TNBCDataset))
    validation_dataset_info = arguments.add_options(parser, "validation_dataset", (None, data.MoNuSegValidationDataset, data.GlaSValidationDataset, data.CoNSePValidationDataset, data.TNBCValidationDataset))
    parser.add_argument("--model_checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--learning_rate", type=float, default=0.0002)

    model_parameters = parse_model_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    trainer = create_trainer(args)

    successfully_started = False
    while not successfully_started:
        model = models.Model(
            counter=models.Counter(args.out_channels, 50, 20),
            similarity_measure=args.loss,
            inter_channel_loss_scaling_factor=1,
            gamma=args.gamma,
            **{parameter: ca.class_type(**ca.arguments) for parameter, ca in model_parameters.items()}
        )

        if args.model_checkpoint is not None:
            model.load_from_checkpoint(args.model_checkpoint)

        train(model, trainer, dataset_info, validation_dataset_info, args)

        successfully_started = os.path.exists(f"{trainer.logger.log_dir}/metrics.csv")
