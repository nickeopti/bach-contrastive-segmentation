import os
import pathlib
import threading
from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import src.data
import src.framework
import src.modules
import src.sampling
import src.similarity
import src.util.arguments
import src.util.events
import src.util.plot
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

AVAILABLE_CONFIDENCE_NETWORKS: Tuple[torch.nn.Module] = (
    src.modules.ConfidenceNetwork,
)
AVAILABLE_FEATURISER_NETWORKS: Tuple[torch.nn.Module] = (
    src.modules.FeatureNetwork,
)


def get_arguments(parser: ArgumentParser):
    train_dataset = src.util.arguments.add_options(parser, 'dataset', list(src.data.available_datasets))
    validation_dataset = src.util.arguments.add_options(parser, 'validation_dataset', [None] + list(src.data.available_validation_datasets))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    sampler = src.util.arguments.add_options_from_module(parser, 'sampler', src.sampling, src.sampling.Sampler)
    similarity_measure = src.util.arguments.add_options_from_module(parser, 'similarity_measure', src.similarity, src.similarity.SimilarityMeasure)

    confidence_network = src.util.arguments.add_options(parser, 'confidence_network', AVAILABLE_CONFIDENCE_NETWORKS)
    featuriser_network = src.util.arguments.add_options(parser, 'featuriser_network', AVAILABLE_FEATURISER_NETWORKS)

    parser.add_argument('--patch_size', type=int, default=50)

    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    return train_dataset, validation_dataset, sampler, similarity_measure, confidence_network, featuriser_network, args


def create_trainer(args) -> pl.Trainer:
    logger = pl.loggers.CSVLogger('logs')

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(
            monitor='best_val_mdc',
            save_top_k=5,
            mode='max',
        )],
    )
    return trainer


def train(model: pl.LightningModule, trainer: pl.Trainer, train_dataset: Dataset, validation_dataset: Dataset = None, batch_size: int = 10, num_workers: int = 1) -> None:
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if validation_dataset:
        validation_data_loader = DataLoader(validation_dataset, batch_size=min(10, len(validation_dataset)), shuffle=False, num_workers=num_workers)

    trainer.fit(model, train_data_loader, validation_data_loader if validation_dataset else None)


def setup_plotting() -> None:
    def plots_path(log_dir: str):
        return f'{log_dir}/plots'

    def plot_region_selection(data: src.framework.LoggedInfo):
        if data.epoch % 5 == 0 and data.batch_idx % 10 == 0:
            to_plot = [
                (
                    image,
                    attention_map,
                    attended_image,
                    [list(map(data.unpack_index, channel_regions)) for channel_regions in positive_regions],
                    [list(map(data.unpack_index, channel_regions)) for channel_regions in negative_regions],
                    data.kernel_size,
                )
                for image, attention_map, attended_image, (positive_regions, negative_regions)
                in zip(data.images.cpu(), data.attention_maps.cpu(), data.attended_images.cpu(), data.sampled_regions)
            ]

            pathlib.Path(plots_path(data.log_dir)).mkdir(parents=True, exist_ok=True)
            threading.Thread(
                target=src.util.plot.plot_selected_crops,
                args=(
                    to_plot,
                    os.path.join(plots_path(data.log_dir), f'selection_{data.epoch}_{data.batch_idx}.png')
                )
            ).start()
    src.util.events.register_event_handler(src.util.events.EventTypes.END_OF_TRAINING_BATCH, plot_region_selection)

    def plot_validation_images(data: src.framework.LoggedInfo):
        if data.epoch % 5 == 0 and data.batch_idx == 0:
            pathlib.Path(plots_path(data.log_dir)).mkdir(parents=True, exist_ok=True)
            threading.Thread(
                target=src.util.plot.plot_mask,
                args=(
                    data.images.cpu(),
                    data.masks.cpu(),
                    data.attention_maps.cpu(),
                    os.path.join(plots_path(data.log_dir), f'validation_{data.epoch}_{data.batch_idx}.png')
                )
            ).start()
    src.util.events.register_event_handler(src.util.events.EventTypes.END_OF_VALIDATION_BATCH, plot_validation_images)


def main() -> None:
    parser = ArgumentParser()
    train_dataset, validation_dataset, sampler, similarity_measure, confidence_network, featuriser_network, args = get_arguments(parser)

    model = src.framework.Model(
        counter=src.framework.Counter(args.out_channels, args.patch_size, 20),
        sampler=sampler,
        similarity_measure=similarity_measure,
        confidence_network=confidence_network,
        featuriser_network=featuriser_network,
        inter_channel_loss_scaling_factor=1,
        learning_rate=args.learning_rate,
        info_to_log=vars(args)
    )

    setup_plotting()

    trainer = create_trainer(args)
    train(model, trainer, train_dataset, validation_dataset, args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
