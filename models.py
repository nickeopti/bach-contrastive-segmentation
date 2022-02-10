import pathlib
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.transforms.transforms import RandomCrop

import plot
from selection import Region, RegionSelector


class Model(pl.LightningModule):
    def __init__(
        self,
        contrastive_selector: RegionSelector,
        attention_network: nn.Module,
        feature_network: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.selector = contrastive_selector
        self.attention_network = attention_network
        self.feature_network = feature_network

        self.cropper = RandomCrop(70)
        self.cos_single = nn.CosineSimilarity(dim=0)
        self.cos_multiple = nn.CosineSimilarity(dim=1)

    def training_step(self, batch, batch_idx):
        to_plot = []

        loss = torch.zeros(1, device=batch.device)

        attention_maps = self.attention_network(batch)
        for image, attention_map in zip(batch, attention_maps):
            attended_image = image * attention_map
            p, n = self.selector.select((attended_image).unsqueeze(0))
            if len(p) < 5 or len(n) < 5:
                continue  # Not enough regions to contrast against each other

            pos = np.random.choice(len(p), 2, replace=False)
            neg = np.random.choice(len(n), min(32, len(n)), replace=False)
            selected_crops: List[Region] = [p[i] for i in pos] + [n[i] for i in neg]

            attended_image_crops = torch.stack(
                [
                    attended_image[region.channel, region.row:region.row + region.size, region.col:region.col + region.size].unsqueeze(0)
                    for region in selected_crops
                ]
            )
            predictions = self.feature_network(attended_image_crops)

            c = self.cos_single(predictions[0], predictions[1])
            cc = self.cos_multiple(predictions[0], predictions[2:])
            contrastive_loss = -torch.log(torch.exp(c) / torch.exp(cc).sum())
            loss += contrastive_loss

            to_plot.append((
                image.detach().cpu(),
                attention_map.detach().cpu(),
                attended_image.detach().cpu(),
                selected_crops,
            ))

        plots_path = f"{self.logger.log_dir}/plots"
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        plot.plot_selected_crops(
            to_plot, path=f"{plots_path}/selection_{self.current_epoch}_{batch_idx}.png"
        )

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.0001)
        return optimizer
