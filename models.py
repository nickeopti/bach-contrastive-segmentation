import pathlib
import threading
from typing import List, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import plot
from selection import Region, Sampler

Image = List
Channel = List
Regions = List[Region]

POSITIVE = 0
NEGATIVE = 1


def sample(population_a: Union[Sequence, torch.Tensor], population_b: Union[Sequence, torch.Tensor], k: int, replace=False):
    n = len(population_a)
    indices = np.random.choice(n, k, replace=replace)
    return [population_a[i] for i in indices], [population_b[i] for i in indices]


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
        return self.f(x) / self.kernel_size ** 2
    
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
        loss: str = "ce",
        learning_rate: float = 0.0002,
        make_histograms: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.counter = counter
        self.sampler = sampler

        self.attention_network = attention_network
        self.feature_network = feature_network

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.inter_channel_loss_scaling_factor = inter_channel_loss_scaling_factor
        self.gamma = gamma

        self.featurise = loss == "feature"
        if loss == "ce":
            def l(x1, x2):
                ce = -(x1 * torch.log2(x2) + (1 - x1) * torch.log2(1 - x2))
                return -ce.mean(dim=(-1, -2))
            self.loss_function = l
        elif loss == "mse":
            def l(x1, x2):
                mse = (x1 - x2) ** 2
                return -mse.mean(dim=(-1, -2))
            self.loss_function = l
        elif loss == "feature":
            self.loss_function = nn.CosineSimilarity(dim=1)

        self.learning_rate = learning_rate

        self.make_histograms = make_histograms

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch
        attention_maps = self.attention_network(images)
        attended_images = images * attention_maps

        n_images = images.shape[0]
        n_classes = attention_maps.shape[1]

        regions, count_shape = self.counter.count(self.sampler.preprocess(attention_maps))
        try:
            sampled_regions = self.sampler.sample(regions, self.counter.count(attention_maps)[0])
        except TypeError:
            sampled_regions = self.sampler.sample(regions)
        # sampled_regions is of shape: image, parity, channel, region

        def unpack_index(idx):
            idx = idx.item()
            row = idx // count_shape[-1]
            col = idx % count_shape[-1]
            return row * self.counter.stride, col * self.counter.stride

        # extract regions
        x = [([], []) for _ in range(n_classes)]
        a = [([], []) for _ in range(n_classes)]
        for image, attention_map, (positive_regions, negative_regions) in zip(images, attention_maps, sampled_regions):
            for parity, parity_regions in [(POSITIVE, positive_regions), (NEGATIVE, negative_regions)]:
                for channel_index, channel_regions in enumerate(parity_regions):
                    x[channel_index][parity].extend((
                        image[0, row:row + self.counter.kernel_size, col:col + self.counter.kernel_size]
                        for row, col in map(unpack_index, channel_regions)
                        if row < image.shape[-2] - self.counter.kernel_size  # disregard sentinels
                    ))
                    a[channel_index][parity].extend((
                        attention_map[channel_index, row:row + self.counter.kernel_size, col:col + self.counter.kernel_size]
                        for row, col in map(unpack_index, channel_regions)
                        if row < attention_map.shape[-2] - self.counter.kernel_size  # disregard sentinels
                    ))

        if any(any(len(p) == 0 for p in c) for c in x):
            return None

        # vectorise within class, parity
        y = [tuple(map(torch.stack, c)) for c in x]
        b = [tuple(map(torch.stack, c)) for c in a]

        if self.featurise:
            # featurise regions; shape class, parity, prediction
            y = [tuple(map(self.feature_network, c)) for c in y]

        def neg(x: torch.Tensor):
            v = 1 - x.mean(dim=(-1, -2))
            e = torch.normal(0, 0.1, v.shape, device=v.device)
            r = torch.maximum(torch.tensor(0, device=x.device), v + e)
            return r
        def pos(x: torch.Tensor):
            v = x.mean(dim=(-1, -2))
            e = torch.normal(0, 0.1, v.shape, device=v.device)
            r = torch.maximum(torch.tensor(0, device=x.device), v + e)
            return r

        # contrast regions:
        loss = torch.zeros(1, device=batch.device)
        for c in range(n_classes):
            # select positive representatives from this class
            positives, a_positives = sample(y[c][POSITIVE], b[c][POSITIVE], n := 10, replace=True)

            # Intra-channel contrastive loss:
            intra_channel_pos_pos_similarity = [
                self.loss_function(positive, y[c][POSITIVE]) * pos(a) * pos(b[c][POSITIVE])
                for positive, a in zip(positives, a_positives)
            ]
            intra_channel_pos_neg_similarity = [
                self.loss_function(positive, y[c][NEGATIVE]) * pos(a) * neg(b[c][NEGATIVE])
                for positive, a in zip(positives, a_positives)
            ]
            intra_channel_contrastive_loss = sum(
                -torch.log(torch.exp(pp) / torch.exp(pn).sum()).sum()
                for pp, pn in zip(intra_channel_pos_pos_similarity, intra_channel_pos_neg_similarity)
            ) / n**2 / n_images
            self.log("intra", intra_channel_contrastive_loss)
            loss += intra_channel_contrastive_loss

            # Inter-channel contrastive loss:
            if n_classes > 1:  # Only makes sense if more than one channel
                # All the positive samples from all the other classes
                all_other_classes_positives = torch.vstack([y[i][POSITIVE] for i in range(len(y)) if i != c])
                a_all_other_classes_positives = torch.vstack([b[i][POSITIVE] for i in range(len(y)) if i != c])

                inter_channel_pos_pos_similarity = [
                    self.loss_function(positive, all_other_classes_positives) * pos(a) * pos(a_all_other_classes_positives)
                    for positive, a in zip(positives, a_positives)
                ]
                inter_channel_contrastive_loss = sum(
                    -torch.log(torch.exp(intra_pp) / torch.exp(inter_pp).sum()).sum()
                    for intra_pp, inter_pp in zip(intra_channel_pos_pos_similarity, inter_channel_pos_pos_similarity)
                ) / n**2 / n_images
                self.log("inter", inter_channel_contrastive_loss)
                loss += self.inter_channel_loss_scaling_factor * inter_channel_contrastive_loss

        # Plotting
        if batch_idx % 10 == 0 and self.current_epoch % 5 == 0:
            to_plot = [
                (
                    image,
                    attention_map,
                    attended_image.detach().cpu(),
                    [list(map(unpack_index, channel_regions)) for channel_regions in positive_regions],
                    [list(map(unpack_index, channel_regions)) for channel_regions in negative_regions],
                    self.counter.kernel_size,
                )
                for image, attention_map, attended_image, (positive_regions, negative_regions)
                in zip(
                    images.detach().cpu(),
                    attention_maps.detach().cpu(),
                    attended_images.detach().cpu(),
                    sampled_regions,
                )
            ]

            plots_path = f"{self.logger.log_dir}/plots"
            pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
            threading.Thread(target=plot.plot_selected_crops, args=(to_plot, f"{plots_path}/selection_{self.current_epoch}_{batch_idx}.png")).start()

            if self.make_histograms:
                to_histogram = [
                    (
                        image.detach().cpu(),
                        attention_map.detach().cpu(),
                        [
                            [region.attention for region in channel]
                            for channel in regions
                        ],
                    )
                    for image, attention_map, regions in zip(images, attention_maps, regions)
                ]
                plot.plot_histograms(to_histogram, f"{plots_path}/histogram_{self.current_epoch}_{batch_idx}.png")
                threading.Thread(target=plot.plot_histograms, args=(to_histogram, f"{plots_path}/histogram_{self.current_epoch}_{batch_idx}.png")).start()

        if loss == 0:
            return None

        regulariser = (0.5 - torch.abs(attention_maps - 0.5)).sum() / torch.tensor(attention_maps.shape[-2:]).prod()
        self.log("reg", regulariser)
        loss += self.gamma * regulariser

        self.log("loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        attention_maps = self.attention_network(images)
        predicted_masks = attention_maps > 0.5

        masks = masks > 0.5  # turn it into a boolean tensor
        tp = torch.logical_and(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)
        fp_fn = torch.logical_xor(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)

        dice_scores = 2 * tp / (2 * tp + fp_fn)

        for image_scores in dice_scores:
            for i, channel_score in enumerate(image_scores, start=1):
                self.log(f"mdc_c{i}", channel_score)

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            plots_path = f"{self.logger.log_dir}/plots"
            pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
            threading.Thread(target=plot.plot_mask, args=(images.cpu(), masks.cpu(), attention_maps.cpu(), f"{plots_path}/validation_{self.current_epoch}_{batch_idx}.png")).start()

        return dice_scores

    def validation_epoch_end(self, validation_step_outputs):
        scores = torch.vstack(validation_step_outputs)
        self.log("best_val_mdc", max(scores.mean(0)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        return optimizer
