from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Collection, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Region:
    row: int
    col: int
    channel: int

    size: int
    attention: float


Regions = List[Region]
Contrastive = Tuple[Regions, Regions]


@dataclass
class ContrastiveRegions:
    positives: Regions
    negatives: Regions


class Filterer:
    def preprocess(self, values: torch.Tensor) -> torch.Tensor:
        return values  # default to noop

    @abstractmethod
    def filter(self, values: List[List[Regions]]) -> List[List[ContrastiveRegions]]:
        pass


class Sampler:
    def __init__(self, n_positives: int = 10, n_negatives: int = 10) -> None:
        self.n_positives = n_positives
        self.n_negatives = n_negatives

    @abstractmethod
    def sample(self, values: List[List[ContrastiveRegions]]) -> Tuple[List[List[Regions]], List[List[Regions]]]:
        pass


class NoFilterer(Filterer):
    def filter(self, values: List[List[Regions]]) -> List[List[ContrastiveRegions]]:
        return [
            [
                ContrastiveRegions(channel, channel)
                for channel in image
            ]
            for image in values
        ]


class SortedFilterer(Filterer):
    def __init__(self, n_positives_filter: int = 50) -> None:
        super().__init__()
        self.n_positives = n_positives_filter

    def filter(self, values: List[List[Regions]]) -> List[List[ContrastiveRegions]]:
        sorted_values = [
            [
                list(sorted(channel, key=lambda region: region.attention, reverse=True))
                for channel in image
            ]
            for image in values
        ]
        return [
            [
                ContrastiveRegions(channel[:self.n_positives], channel[self.n_positives:])
                for channel in image
            ]
            for image in sorted_values
        ]


class UniformSampler(Sampler):
    def sample(self, values: List[List[ContrastiveRegions]]) -> Tuple[List[List[Regions]], List[List[Regions]]]:
        # res = [[None] * len(values)]
        # for image_index, image in enumerate(values):
        #     for channel_index, channel in enumerate(image):
        #         positive_indices = np.random.choice(self.n_positives, replace=False)
        #         negative_indices = np.random.choice(self.n_negatives, replace=False)

        #         positives = [channel.positives[i] for i in positive_indices]
        #         negatives = [channel.negatives[i] for i in negative_indices]

        #         res[image_index][channel_index] = ContrastiveRegions(positives, negatives)

        positives = [
            [
                [channel.positives[i] for i in np.random.choice(len(channel.positives), self.n_positives, replace=False)]
                for channel in image
            ]
            for image in values
        ]
        negatives = [
            [
                [channel.negatives[i] for i in np.random.choice(len(channel.negatives), self.n_negatives, replace=False)]
                for channel in image
            ]
            for image in values
        ]

        return positives, negatives


class RegionSelector:
    @abstractmethod
    def select(self, attended_image) -> Sequence[Collection[Region]]:
        pass


class AttentionBasedRegionSelector(RegionSelector):
    def __init__(self, size: int, stride: int, device=None) -> None:
        self.size = size
        self.stride = stride

        self.f = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=self.size, stride=self.stride, bias=False
        )
        self.f.weight = nn.Parameter(
            torch.ones((1, 1, self.size, self.size), dtype=torch.float, device=device),
            requires_grad=False,
        )

    def transform(self, attentions, where):
        *_, channel, row, col = where
        return [
            Region(x * self.stride, y * self.stride, c, self.size, attentions[0, c, x, y] if attentions is not None else 0)
            for c, x, y in zip(channel.tolist(), row.tolist(), col.tolist())  # implicit cast to floats, rather than tensors containing floats
        ]


class ContrastiveRegionsQuantileSelector(AttentionBasedRegionSelector):
    def __init__(self, size: int, stride: int, quantile: float = 0.9, device=None) -> None:
        super().__init__(size, stride, device)
        self.quantile = quantile


class SingleChannelQuantileSelector(ContrastiveRegionsQuantileSelector):
    def __init__(self, size: int = 50, stride: int = 5, quantile: float = 0.9, device=None, *vargs, **kwargs) -> None:
        super().__init__(size, stride, quantile, device)

    def select(self, attended_image):
        threshold = torch.quantile(attended_image, 0.9)
        activated = (attended_image > threshold).type(torch.float)
        slide_sum = self.f(activated)

        positives = self.transform(slide_sum, torch.where(slide_sum > self.size**2 / 2))
        negatives = self.transform(slide_sum, torch.where(slide_sum < self.size**2 / 2))
        return positives, negatives


class MultiChannelQuantileSelector(ContrastiveRegionsQuantileSelector):
    def __init__(self, size: int = 50, stride: int = 5, quantile: float = 0.9, device=None) -> None:
        super().__init__(size, stride, quantile, device)

    def select(self, attended_image):
        attentions = torch.stack([
            (self.f((channel > torch.quantile(channel, self.quantile)).type(torch.float).unsqueeze(0).unsqueeze(0))).squeeze()
            for channel in attended_image[0]
        ]).unsqueeze(0)  # assuming batch size of 1
        l = self.transform(attentions, torch.where(attentions > self.size ** 2 / 2))
        return [
            [region for region in l if region.channel == i]
            for i in range(len(attended_image[0]))
        ]


class UniformRandomSelector(RegionSelector):
    def __init__(self, size: int = 50, n_positives: int = 5, n_negatives: int = 32, device=None) -> None:
        super().__init__()
        self.size = size
        self.n_positives = n_positives
        self.n_negatives = n_negatives
    
    def select(self, attended_image):
        *_, width, height = attended_image.shape
        positives_rows = torch.randint(width - self.size, (self.n_positives,))
        positives_cols = torch.randint(height - self.size, (self.n_positives,))
        negatives_rows = torch.randint(width - self.size, (self.n_negatives,))
        negatives_cols = torch.randint(height - self.size, (self.n_negatives,))
        
        positives = [
            Region(row, col, 0, self.size, 0)
            for row, col in zip(positives_rows, positives_cols)
        ]
        negatives = [
            Region(row, col, 0, self.size, 0)
            for row, col in zip(negatives_rows, negatives_cols)
        ]
        return positives, negatives


class SingleChannelSortedAttentionSelector(AttentionBasedRegionSelector):
    def __init__(self, size: int = 50, stride: int = 20, n_positives: int = 30, device=None) -> None:
        super().__init__(size, stride, device)
        self.n_positives = n_positives

    def select(self, attended_image) -> Sequence[Collection[Region]]:
        attentions = self.f(attended_image)
        regions = self.transform(attentions, torch.where(torch.ones_like(attentions)))
        sorted_regions = list(sorted(regions, key=lambda region: region.attention, reverse=True))
        return sorted_regions[:self.n_positives], sorted_regions[self.n_positives:]


class MultiChannelSortedAttentionSelector(AttentionBasedRegionSelector):
    def __init__(self, size: int = 50, stride: int = 20, n_positives: int = 30, device=None) -> None:
        super().__init__(size, stride, device)
        self.n_positives = n_positives

    def select(self, attended_image) -> Sequence[Collection[Region]]:
        channel_attentions = [self.f(channel.unsqueeze(0).unsqueeze(0)) for channel in attended_image[0]]  # Assuming batch size of 1
        channel_regions = [self.transform(attention, torch.where(torch.ones_like(attention))) for attention in channel_attentions]
        channel_sorted_regions = [list(sorted(regions, key=lambda region: region.attention, reverse=True)) for regions in channel_regions]
        return [sorted_regions[:self.n_positives] for sorted_regions in channel_sorted_regions]
