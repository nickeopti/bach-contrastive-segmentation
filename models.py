import itertools
import pathlib
import threading
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import plot
from selection import ContrastiveRegions, Filterer, Region, RegionSelector, Sampler


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
    
    def count(self, x: torch.Tensor) -> List[List[List[Region]]]:
        counts = self(x)
        image_counts = [
            [
                [
                    [
                        Region(
                            row=row_index * self.stride,
                            col=column_index * self.stride,
                            channel=channel_index,
                            size=self.kernel_size,
                            attention=element
                        )
                        for column_index, element in enumerate(row)
                    ]
                    for row_index, row in enumerate(channel)
                ]
                for channel_index, channel in enumerate(image)
            ]
            for image in counts
        ]
        images_regions = [[list(itertools.chain(*channel)) for channel in image] for image in image_counts]
        return images_regions
    
    def transform_to_regions(self, regions: torch.Tensor, where: ContrastiveRegions) -> ContrastiveRegions:
        pass


class Model(pl.LightningModule):
    def __init__(
        self,
        counter: Counter,
        filterer: Filterer,
        sampler: Sampler,
        attention_network: nn.Module,
        feature_network: nn.Module,
        inter_channel_loss_scaling_factor: float = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.counter = counter
        self.filterer = filterer
        self.sampler = sampler

        self.attention_network = attention_network
        self.feature_network = feature_network

        self.cos_single = nn.CosineSimilarity(dim=0)
        self.cos_multiple = nn.CosineSimilarity(dim=1)
        self.inter_channel_loss_scaling_factor = inter_channel_loss_scaling_factor

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch
        attention_maps = self.attention_network(images)
        attended_images = images * attention_maps

        attentions = self.filterer.preprocess(attended_images)
        regions = self.counter.count(attentions)
        filtered_regions = self.filterer.filter(regions)

        sampled_positive_regions, \
        sampled_negative_regions = self.sampler.sample(filtered_regions)

        loss = torch.zeros(1, device=batch.device)
        for attended_image, positive_regions, negative_regions in zip(attended_images, sampled_positive_regions, sampled_negative_regions):
            if any(len(r) < 2 for r in positive_regions + negative_regions):
                continue  # Not enough regions to contrast against each other

            positive_crops, negative_crops = (
                torch.stack(list(itertools.chain(
                    *[
                        [
                            attended_image[region.channel, region.row:region.row + region.size, region.col:region.col + region.size].unsqueeze(0)
                            for region in regions
                        ]
                        for regions in region_type
                    ]
                )))
                for region_type in (positive_regions, negative_regions)
            )
            positive_features = self.feature_network(positive_crops)
            negative_features = self.feature_network(negative_crops)

            positive_lengths = [len(r) for r in positive_regions]
            negative_lengths = [len(r) for r in negative_regions]
            positive_class_to_index_range = [(sum(positive_lengths[:i]), sum(positive_lengths[:i+1])) for i in range(len(positive_regions))]
            negative_class_to_index_range = [(sum(negative_lengths[:i]), sum(negative_lengths[:i+1])) for i in range(len(negative_regions))]

            for (positive_from_ci, positive_to_ci), (negative_from_ci, negative_to_ci) in zip(positive_class_to_index_range, negative_class_to_index_range):
                # Select two positive representatives from this class
                i1, i2 = np.random.choice(positive_to_ci - positive_from_ci, 2, replace=False)
                pos_1 = positive_features[positive_from_ci + i1]
                pos_2 = positive_features[positive_from_ci + i2]

                # Intra-channel contrastive loss:
                intra_channel_pos_pos_similarity = self.cos_single(pos_1, pos_2)
                intra_channel_pos_neg_similarity = self.cos_multiple(pos_1, negative_features[negative_from_ci:negative_to_ci])
                intra_channel_contrastive_loss = -torch.log(torch.exp(intra_channel_pos_pos_similarity) / torch.exp(intra_channel_pos_neg_similarity).sum())

                # All the positive samples from all the other classes
                all_other_classes_positives = torch.vstack((positive_features[:positive_from_ci], positive_features[positive_to_ci:]))
                inter_channel_pos_pos_similarity = self.cos_multiple(pos_1, all_other_classes_positives)
                inter_channel_contrastive_loss = -torch.log(torch.exp(intra_channel_pos_pos_similarity) / torch.exp(inter_channel_pos_pos_similarity).sum())

                loss += intra_channel_contrastive_loss + self.inter_channel_loss_scaling_factor * inter_channel_contrastive_loss


            # raise RuntimeError()


        # to_plot.append((
        #         image.detach().cpu(),
        #         attention_map.detach().cpu(),
        #         attended_image.detach().cpu(),
        #         selected_classes,
        #         class_indices,
        #     ))
        to_plot = [
            (
                image.detach().cpu(),
                attention_map.detach().cpu(),
                attended_image.detach().cpu(),
                positive_regions,
                negative_regions,
            )
            for image, attention_map, attended_image, positive_regions, negative_regions
            in zip(images, attention_maps, attended_images, sampled_positive_regions, sampled_negative_regions)
        ]





        # to_plot = []

        # loss = torch.zeros(1, device=batch.device)


        # for image, attention_map, attended_image in zip(batch, attention_maps, attended_images):
        #     # attended_image = image * attention_map

        #     classes = self.selector.select((attended_image).unsqueeze(0))
        #     if any(len(c) < 2 for c in classes):
        #         continue  # Not enough regions to contrast against each other

        #     n_chosen_regions_in_each_class = 10
        #     class_indices = [min(n_chosen_regions_in_each_class, len(c)) for c in classes]
        #     class_indices = [(sum(class_indices[:i]), sum(class_indices[:i+1])) for i in range(len(class_indices))]

        #     t = [
        #         [c[i] for i in np.random.choice(len(c), min(n_chosen_regions_in_each_class, len(c)), replace=False)]
        #         for c in classes
        #     ]
        #     selected_classes = list(itertools.chain(*t))

        #     attended_image_crops = torch.stack(
        #         [
        #             attended_image[region.channel, region.row:region.row + region.size, region.col:region.col + region.size].unsqueeze(0)
        #             for region in selected_classes
        #         ]
        #     )
        #     predictions = self.feature_network(attended_image_crops)

        #     for ci_from, ci_to in class_indices:
        #         i1, i2 = np.random.choice(ci_to-ci_from, 2, replace=False)
        #         p1 = predictions[ci_from+i1]
        #         p2 = predictions[ci_from+i2]
        #         c = self.cos_single(p1, p2)
        #         rest = torch.vstack((predictions[:ci_from], predictions[ci_to:]))
        #         cc = self.cos_multiple(p1, rest)
        #         contrastive_loss = -torch.log(torch.exp(c) / torch.exp(cc).sum())
        #         loss += contrastive_loss

        #     to_plot.append((
        #         image.detach().cpu(),
        #         attention_map.detach().cpu(),
        #         attended_image.detach().cpu(),
        #         selected_classes,
        #         class_indices,
        #     ))

        plots_path = f"{self.logger.log_dir}/plots"
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        plot_thread = threading.Thread(target=plot.plot_selected_crops, args=(to_plot, f"{plots_path}/selection_{self.current_epoch}_{batch_idx}.png"))
        plot_thread.start()

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.0001)
        return optimizer
