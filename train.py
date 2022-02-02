import os.path
import pathlib
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomCrop

import data
import model
import plot


def contrastive_areas(attended_image, size, stride=5):
    f = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=stride, bias=False)
    f.weight = nn.Parameter(torch.ones((1, 1, size, size), dtype=torch.float, device=attended_image.device), requires_grad=False)

    def positive_indices(channel_index, where_indices):
        _, _, row, col = where_indices
        return list(zip(
            [channel_index]*row.numel(),
            (row*stride).tolist(),
            (col*stride).tolist(),
            [size]*row.numel()
        ))

    return [
        positive_indices(i, torch.where(f((channel > torch.quantile(channel, 0.9)).type(torch.float).unsqueeze(0).unsqueeze(0)) > size**2 / 2))
        for i, channel in enumerate(attended_image[0])  # Assuming batch size of 1
    ]

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.attention_network = model.AttentionNetwork(1, 8, 2)
        self.classifier_network = model.TypeNetwork(1)
        self.cropper = RandomCrop(70)
        self.cos_single = nn.CosineSimilarity(dim=0)
        self.cos_multiple = nn.CosineSimilarity(dim=1)

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=batch.device)

        attention_maps = self.attention_network(batch)
        to_plot = []
        for image, attention_map in zip(batch, attention_maps):
            attended_image = (image * attention_map)
            p, n = contrastive_areas((attended_image).unsqueeze(0), 50, 20)
            if len(p) >= 5 and len(n) >= 5:
                pos = np.random.choice(len(p), 2, replace=False)
                neg = np.random.choice(len(n), min(32, len(n)), replace=False)
                selected_crops = [p[i] for i in pos] + [n[i] for i in neg]

                if len(to_plot) < 10:
                    to_plot.append((
                        image.detach().cpu(),
                        attention_map.squeeze().detach().cpu(),
                        attended_image.detach().cpu(),
                        selected_crops
                    ))

                cropped_images = torch.stack([image.squeeze()[row:row+size, col:col+size].unsqueeze(0) for _, row, col, size in selected_crops])
                cropped_attenmaps = torch.stack([attention_map.squeeze()[channel, row:row+size, col:col+size].unsqueeze(0) for channel, row, col, size in selected_crops])

                predictions = self.classifier_network(cropped_images * cropped_attenmaps)
                c = self.cos_single(predictions[0], predictions[1])
                cc = self.cos_multiple(predictions[0], predictions[2:])
                contrastive_loss = -torch.log(torch.exp(c) / torch.exp(cc).sum())
                loss += contrastive_loss

        plots_path = f'{self.logger.log_dir}/plots'
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        plot.plot_selected_crops(to_plot, path=f'{plots_path}/selection_{self.current_epoch}_{batch_idx}.png')

        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.0001)
        return optimizer


def train(epochs, model_checkpoint_path=None):
    dataset = data.SamplesDataset('image.tiff')

    train_data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    network = Model()
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
