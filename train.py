import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import data
import models
import selection


def train(args):
    dataset = data.SamplesDataset(args.dataset)
    train_data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    network = models.Model(selection.contrastive_areas)
    if args.model_checkpoint is not None:
        network.load_from_checkpoint(args.model_checkpoint)

    logger = pl.loggers.CSVLogger("logs")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(every_n_epochs=1)],
    )

    try:
        trainer.fit(network, train_data_loader)
    except RuntimeError as exc:
        if "element 0 of tensors does not require grad" not in str(exc):
            raise exc

    return logger.log_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="image.tiff")
    parser.add_argument("--model_checkpoint", type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    successfully_started = False
    while not successfully_started:
        log_dir = train(args)
        successfully_started = os.path.exists(f"{log_dir}/metrics.csv")
