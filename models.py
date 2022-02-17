import itertools
import pathlib
import threading
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

            classes = self.selector.select((attended_image).unsqueeze(0))
            if any(len(c) < 2 for c in classes):
                continue  # Not enough regions to contrast against each other

            n_chosen_regions_in_each_class = 10
            class_indices = [min(n_chosen_regions_in_each_class, len(c)) for c in classes]
            class_indices = [(sum(class_indices[:i]), sum(class_indices[:i+1])) for i in range(len(class_indices))]

            t = [
                [c[i] for i in np.random.choice(len(c), min(n_chosen_regions_in_each_class, len(c)), replace=False)]
                for c in classes
            ]
            selected_classes = list(itertools.chain(*t))

            attended_image_crops = torch.stack(
                [
                    attended_image[region.channel, region.row:region.row + region.size, region.col:region.col + region.size].unsqueeze(0)
                    for region in selected_classes
                ]
            )
            predictions = self.feature_network(attended_image_crops)

            for ci_from, ci_to in class_indices:
                i1, i2 = np.random.choice(ci_to-ci_from, 2, replace=False)
                p1 = predictions[ci_from+i1]
                p2 = predictions[ci_from+i2]
                c = self.cos_single(p1, p2)
                rest = torch.vstack((predictions[:ci_from], predictions[ci_to:]))
                cc = self.cos_multiple(p1, rest)
                contrastive_loss = -torch.log(torch.exp(c) / torch.exp(cc).sum())
                loss += contrastive_loss

            to_plot.append((
                image.detach().cpu(),
                attention_map.detach().cpu(),
                attended_image.detach().cpu(),
                selected_classes,
                class_indices,
            ))

        plots_path = f"{self.logger.log_dir}/plots"
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        plot_thread = threading.Thread(target=plot.plot_selected_crops, args=(to_plot, f"{plots_path}/selection_{self.current_epoch}_{batch_idx}.png"))
        plot_thread.start()

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.0001)
        return optimizer
