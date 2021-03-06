import pathlib
import threading
from typing import List, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.transforms.transforms import Grayscale
from kornia.enhance.histogram import histogram2d

import plot
from selection import Region, Sampler

Image = List
Channel = List
Regions = List[Region]

POSITIVE = 0
NEGATIVE = 1


def sample(population: Union[Sequence, torch.Tensor], k: int, replace=False):
    n = len(population)
    indices = np.random.choice(n, k, replace=replace)
    return [population[i] for i in indices]


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
        similarity_measure: str = "ce",
        learning_rate: float = 0.0002,
        make_histograms: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['counter', 'attention_network', 'feature_network'])

        self.counter = counter
        self.sampler = sampler

        self.attention_network = attention_network
        self.feature_network = feature_network

        self.inter_channel_loss_scaling_factor = inter_channel_loss_scaling_factor
        self.gamma = gamma

        self.featurise = similarity_measure == "feature"
        if similarity_measure == "ce":
            def similarity(x1: torch.Tensor, x2: torch.Tensor):
                assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
                assert len(x1.shape) == 3
                ce = -(x1 * torch.log2(x2) + (1 - x1) * torch.log2(1 - x2))
                return -ce.mean(dim=(-1, -2, -3))
            self.similarity_measure = similarity
        elif similarity_measure == "mse":
            def similarity(x1: torch.Tensor, x2: torch.Tensor):
                assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
                assert len(x1.shape) == 3
                mse = (x1 - x2) ** 2
                return -mse.mean(dim=(-1, -2, -3))
            self.similarity_measure = similarity
        elif similarity_measure == "feature":
            cosine_similarity = nn.CosineSimilarity(dim=1)
            def similarity(x1: torch.Tensor, x2: torch.Tensor):
                assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
                assert len(x1.shape) == 1
                return cosine_similarity(x1, x2)
            self.similarity_measure = similarity
        elif similarity_measure == "mi":
            grey_scale = Grayscale(1)
            def similarity(x1: torch.Tensor, x2: torch.Tensor):
                assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
                assert len(x1.shape) == 3
                if len(x2.shape) == 3:
                    x2 = x2.unsqueeze(0)
                def mi(v1, v2):
                    v1 = grey_scale(v1).flatten(start_dim=1)
                    v2 = grey_scale(v2).flatten(start_dim=1)
                    joint_histogram = histogram2d(v1, v2, bins=torch.linspace(0, 1, 25, device=v1.device), bandwidth=torch.tensor(0.9))
                    p_xy = joint_histogram / joint_histogram.sum()
                    p_x = p_xy.sum(dim=2)
                    p_y = p_xy.sum(dim=1)
                    p_x_p_y = p_x[:, :, None] * p_y[:, None, :]
                    mi = (p_xy * torch.log2(p_xy / p_x_p_y)).sum(dim=(1, 2))
                    return mi
                b = x2.shape[0]
                x1 = x1.repeat(b, 1, 1).reshape(b, *x1.shape)
                return mi(x1, x2)
            self.similarity_measure = similarity
        elif similarity_measure == "kl":
            grey_scale = Grayscale(1)
            kl_divergence = nn.KLDivLoss(reduction="batchmean", log_target=True)
            def similarity(x1, x2):
                assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
                assert len(x1.shape) == 3
                if len(x2.shape) == 3:
                    x2 = x2.unsqueeze(0)
                v1 = torch.maximum(grey_scale(x1).flatten().log(), torch.tensor(-100))
                v2 = torch.maximum(grey_scale(x2).flatten(start_dim=1).log(), torch.tensor(-100))
                divergence = torch.stack([kl_divergence(v1, v) for v in v2])
                return -divergence
            self.similarity_measure = similarity
        else:
            raise RuntimeError(f"Unknown similarity measure {similarity_measure!r}")

        self.learning_rate = learning_rate

        self.make_histograms = make_histograms

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch
        attention_maps = self.attention_network(images)
        attended_images = torch.stack([images[:, i, None] * attention_maps for i in range(images.shape[1])], dim=2)

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
        for attended_image, (positive_regions, negative_regions) in zip(attended_images, sampled_regions):
            for parity, parity_regions in [(POSITIVE, positive_regions), (NEGATIVE, negative_regions)]:
                for channel_index, channel_regions in enumerate(parity_regions):
                    x[channel_index][parity].extend((
                        attended_image[channel_index, :, row:row + self.counter.kernel_size, col:col + self.counter.kernel_size]
                        for row, col in map(unpack_index, channel_regions)
                        if row < attended_image.shape[-2] - self.counter.kernel_size  # disregard sentinels
                    ))

        if any(any(len(p) == 0 for p in c) for c in x):
            return None

        # vectorise within class, parity
        y = [tuple(map(torch.stack, c)) for c in x]

        if self.featurise:
            # featurise regions; shape class, parity, prediction
            y = [tuple(map(self.feature_network, c)) for c in y]

        # contrast regions:
        loss = torch.zeros(1, device=batch.device)
        for c in range(n_classes):
            # select positive representatives from this class
            positives = sample(y[c][POSITIVE], n := 10, replace=True)

            # Intra-channel contrastive loss:
            intra_channel_pos_pos_similarity = [
                self.similarity_measure(positive, y[c][POSITIVE])
                for positive in positives
            ]
            intra_channel_pos_neg_similarity = [
                self.similarity_measure(positive, y[c][NEGATIVE])
                for positive in positives
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

                inter_channel_pos_pos_similarity = [
                    self.similarity_measure(positive, all_other_classes_positives)
                    for positive in positives
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
        with torch.no_grad():
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
