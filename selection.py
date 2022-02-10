from abc import abstractmethod
from dataclasses import dataclass
from typing import Collection, Sequence

import torch
import torch.nn as nn


@dataclass
class Region:
    row: int
    col: int
    channel: int

    size: int
    attention: float


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
        attentions = torch.stack([(self.f((channel > torch.quantile(channel, self.quantile)).type(torch.float).unsqueeze(0).unsqueeze(0))).squeeze() for channel in attended_image[0]]).unsqueeze(0)  # assuming batch size of 1
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
