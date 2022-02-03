import os.path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import data
import models
import selection


def train(epochs, model_checkpoint_path=None):
    dataset = data.SamplesDataset('image.tiff')

    train_data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    network = models.Model(selection.contrastive_areas)
    if model_checkpoint_path is not None:
        network.load_from_checkpoint(model_checkpoint_path)

    logger = pl.loggers.CSVLogger('logs')

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(every_n_epochs=1)
        ],
        gpus=1
    )

    try:
        trainer.fit(network, train_data_loader)
    except RuntimeError as exc:
        if not 'element 0 of tensors does not require grad' in str(exc):
            raise exc

    return logger.log_dir


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    successfully_started = False
    while not successfully_started:
        log_dir = train(epochs, model_path)
        successfully_started = os.path.exists(f'{log_dir}/metrics.csv')
