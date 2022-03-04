import pathlib
import threading
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import plot
from selection import Region, Sampler

Image = List
Channel = List
Regions = List[Region]


class Counter(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.f = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, bias=False)
        w = torch.stack(
            [
                torch.stack(
                    [
                        (torch.ones if i == j else torch.zeros)(kernel_size, kernel_size, dtype=torch.float)
                        for j in range(channels)
                    ]
                )
                for i in range(channels)
            ]
        )
        self.f.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
    
    def count(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        counts: torch.Tensor = self(x)
        return counts.flatten(start_dim=-2), counts.shape


class Model(pl.LightningModule):
    def __init__(
        self,
        counter: Counter,
        sampler: Sampler,
        attention_network: nn.Module,
        feature_network: nn.Module,
        inter_channel_loss_scaling_factor: float = 1,
        gamma: float = 1,
        make_histograms: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.counter = counter
        self.sampler = sampler

        self.attention_network = attention_network
        self.feature_network = feature_network

        self.cos_single = nn.CosineSimilarity(dim=0)
        self.cos_multiple = nn.CosineSimilarity(dim=1)
        self.inter_channel_loss_scaling_factor = inter_channel_loss_scaling_factor
        self.gamma = gamma

        self.make_histograms = make_histograms

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch
        attention_maps = self.attention_network(images)
        attended_images = images * attention_maps

        regions, count_shape = self.counter.count(attended_images)
        sampled_regions = self.sampler.sample(regions, self.counter.kernel_size ** 2)
        sampled_regions_rows: torch.Tensor = torch.div(sampled_regions, count_shape[-1], rounding_mode='trunc') * self.counter.stride
        sampled_regions_cols: torch.Tensor = (sampled_regions % count_shape[-1]) * self.counter.stride

        loss = torch.zeros(1, device=batch.device)
        for attended_image, (positive_regions_rows, negative_regions_rows), (positive_regions_cols, negative_regions_cols) in zip(attended_images, sampled_regions_rows, sampled_regions_cols):
            positive_lengths = [len([row for row in r if row < attended_image.shape[-2] - self.counter.kernel_size]) for r in positive_regions_rows]
            negative_lengths = [len([row for row in r if row < attended_image.shape[-2] - self.counter.kernel_size]) for r in negative_regions_rows]

            if any(l < 2 for l in positive_lengths) or any(l < 2 for l in negative_lengths):
                continue  # Not enough regions to contrast against each other

            positive_crops, negative_crops = (
                torch.vstack(
                    [
                        torch.stack([
                            attended_image[channel_index, row:row + self.counter.kernel_size, col:col + self.counter.kernel_size].unsqueeze(0)
                            for row, col in zip(channel_rows, channel_cols) if row < attended_image.shape[-2] - self.counter.kernel_size
                        ])
                        for channel_index, (channel_rows, channel_cols) in enumerate(region_type)
                    ]
                )
                for region_type in (zip(positive_regions_rows, positive_regions_cols), zip(negative_regions_rows, negative_regions_cols))
            )
            positive_features = self.feature_network(positive_crops)
            negative_features = self.feature_network(negative_crops)

            positive_class_to_index_range = [(sum(positive_lengths[:i]), sum(positive_lengths[:i+1])) for i in range(len(positive_regions_rows))]
            negative_class_to_index_range = [(sum(negative_lengths[:i]), sum(negative_lengths[:i+1])) for i in range(len(negative_regions_rows))]

            for (positive_from_ci, positive_to_ci), (negative_from_ci, negative_to_ci) in zip(positive_class_to_index_range, negative_class_to_index_range):
                # Select two positive representatives from this class
                i1, i2 = np.random.choice(positive_to_ci - positive_from_ci, 2, replace=False)
                pos_1 = positive_features[positive_from_ci + i1]
                pos_2 = positive_features[positive_from_ci + i2]

                # Intra-channel contrastive loss:
                intra_channel_pos_pos_similarity = self.cos_single(pos_1, pos_2)
                intra_channel_pos_neg_similarity = self.cos_multiple(pos_1, negative_features[negative_from_ci:negative_to_ci])
                intra_channel_contrastive_loss = -torch.log(torch.exp(intra_channel_pos_pos_similarity) / torch.exp(intra_channel_pos_neg_similarity).sum())
                self.log("intra", intra_channel_contrastive_loss)
                loss += intra_channel_contrastive_loss

                # Inter-channel contrastive loss:
                if len(positive_class_to_index_range) > 1:  # Only makes sense if more than one channel
                    # All the positive samples from all the other classes
                    all_other_classes_positives = torch.vstack((positive_features[:positive_from_ci], positive_features[positive_to_ci:]))

                    inter_channel_pos_pos_similarity = self.cos_multiple(pos_1, all_other_classes_positives)
                    inter_channel_contrastive_loss = -torch.log(torch.exp(intra_channel_pos_pos_similarity) / torch.exp(inter_channel_pos_pos_similarity).sum())
                    self.log("inter", inter_channel_contrastive_loss)
                    loss += self.inter_channel_loss_scaling_factor * inter_channel_contrastive_loss

        if batch_idx % 10 == 0 and self.current_epoch % 5 == 0:
            to_plot = [
                (
                    image,
                    attention_map,
                    attended_image.detach().cpu(),
                    [list(zip(rows, cols)) for rows, cols in zip(regions_rows[0], regions_cols[0])],  # positive regions
                    [list(zip(rows, cols)) for rows, cols in zip(regions_rows[1], regions_cols[1])],  # negative regions
                    self.counter.kernel_size,
                )
                for image, attention_map, attended_image, regions_rows, regions_cols
                in zip(
                    images.detach().cpu(),
                    attention_maps.detach().cpu(),
                    attended_images.detach().cpu(),
                    sampled_regions_rows.detach().cpu(),
                    sampled_regions_cols.detach().cpu(),
                )
            ]

            plots_path = f"{self.logger.log_dir}/plots"
            pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

            if self.make_histograms:
                to_histogram = [
                    (
                        image.detach().cpu(),
                        attended_image.detach().cpu(),
                        [
                            [region.attention for region in channel]
                            for channel in regions
                        ],
                    )
                    for image, attended_image, regions in zip(images, attended_images, regions)
                ]
                plot.plot_histograms(to_histogram, f"{plots_path}/histogram_{self.current_epoch}_{batch_idx}.png")

            plot_thread = threading.Thread(target=plot.plot_selected_crops, args=(to_plot, f"{plots_path}/selection_{self.current_epoch}_{batch_idx}.png"))
            plot_thread.start()

        if loss == 0:
            raise RuntimeError

        regulariser = (0.5 - torch.abs(attention_maps - 0.5)).sum() / torch.tensor(attention_maps.shape[-2:]).prod()
        self.log("reg", regulariser)
        loss += self.gamma * regulariser

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.0001)
        return optimizer
