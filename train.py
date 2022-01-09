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


def contrastive_areas(attention_map, size, stride=5):
    f = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=stride, bias=False)
    f.weight = nn.Parameter(torch.ones((1, 1, size, size), dtype=torch.int), requires_grad=False)

    threshold = torch.quantile(attention_map, 0.9)
    activated = (attention_map > threshold).type(torch.int)
    slide_sum = f(activated)

    _, _, rows, cols = torch.where(slide_sum > size**2 / 2)
    positives = list(zip((rows*stride).tolist(), (cols*stride).tolist(), [size]*rows.numel()))

    _, _, rows, cols = torch.where(slide_sum < size**2 / 2)
    negatives = list(zip((rows*stride).tolist(), (cols*stride).tolist(), [size]*rows.numel()))

    return positives, negatives


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.attention_network = model.AttentionNetwork(1, 8)
        self.classifier_network = model.TypeNetwork(1)
        self.cropper = RandomCrop(70)
        self.cos_single = nn.CosineSimilarity(dim=0)
        self.cos_multiple = nn.CosineSimilarity(dim=1)

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1)

        attention_maps = self.attention_network(batch)
        to_plot = []
        for image, attention_map in zip(batch, attention_maps):
            p, n = contrastive_areas((image * attention_map).unsqueeze(0), 50, 20)
            if len(p) >= 2 and len(n) >= 5:
                pos = np.random.choice(len(p), 2, replace=False)
                neg = np.random.choice(len(n), 32, replace=False)
                selected_crops = [p[i] for i in pos] + [n[i] for i in neg]

                if len(to_plot) < 10:
                    to_plot.append((image.detach(), attention_map.detach(), selected_crops))

                cropped_images = torch.stack([image.squeeze()[row:row+size, col:col+size].unsqueeze(0) for row, col, size in selected_crops])
                cropped_attenmaps = torch.stack([attention_map.squeeze()[row:row+size, col:col+size].unsqueeze(0) for row, col, size in selected_crops])

                predictions = self.classifier_network(cropped_images * cropped_attenmaps)
                c = self.cos_single(predictions[0], predictions[1])
                cc = self.cos_multiple(predictions[0], predictions[2:])
                contrastive_loss = -torch.log(torch.exp(c) / torch.exp(cc).sum())
                loss += contrastive_loss

        plots_path = f'{self.logger.log_dir}/plots'
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        plot.plot_selected_crops(to_plot, path=f'{plots_path}/selection_{self.current_epoch}_{batch_idx}.png')

        self.log('loss', loss, on_step=True)

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

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=pl.loggers.CSVLogger('logs'),
        callbacks=[
            ModelCheckpoint(every_n_epochs=1)
        ]
    )
    trainer.fit(network, train_data_loader)


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    train(epochs, model_path)
