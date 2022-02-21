from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


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


class ThresholdFilterer(Filterer):
    def __init__(self, threshold: float = 0.5, normalised: bool = True) -> None:
        self.threshold = threshold
        self.normalised = normalised

    def filter(self, values: List[List[Regions]]) -> List[List[ContrastiveRegions]]:
        normalisation_factor = values[0][0][0].size ** 2 if self.normalised else 1
        return [
            [
                ContrastiveRegions(
                    [region for region in channel if region.attention > self.threshold * normalisation_factor],
                    [region for region in channel if region.attention < self.threshold * normalisation_factor]
                )
                for channel in image
            ]
            for image in values
        ]


class QuantileFilterer(ThresholdFilterer):
    def __init__(self, threshold: float = 0.5, normalised: bool = True, quantile: float = 0.9) -> None:
        super().__init__(threshold, normalised)
        self.quantile = quantile

    def preprocess(self, values: torch.Tensor) -> torch.Tensor:
        b, c, *_ = values.shape
        quantiles = [
            torch.quantile(image.flatten(start_dim=-2), q=self.quantile, dim=-1)
            for image in values
        ]

        result = values.clone()
        for i in range(b):
            for j in range(c):
                result[i, j] = (values[i, j] >= quantiles[i][j]).type(values.type())
        return result


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
        positives = [
            [
                [
                    channel.positives[i] for i in (
                        np.random.choice(len(channel.positives), self.n_positives, replace=True)
                        if len(channel.positives) > 0 else []
                    )
                ]
                for channel in image
            ]
            for image in values
        ]
        negatives = [
            [
                [
                    channel.negatives[i] for i in (
                        np.random.choice(len(channel.negatives), self.n_negatives, replace=True)
                        if len(channel.negatives) else []
                    )
                ]
                for channel in image
            ]
            for image in values
        ]

        return positives, negatives


class ProbabilisticSampler(Sampler):
    def sample(self, values: List[List[ContrastiveRegions]]) -> Tuple[List[List[Regions]], List[List[Regions]]]:
        # Currently assumes that positives and negatives in each ContrastiveRegions are identical
        activations = [
            [
                np.array([region.attention for region in channel.positives])
                for channel in image
            ]
            for image in values
        ]
        sums = [
            [
                channel.sum()
                for channel in image
            ]
            for image in activations
        ]
        positive_probabilities = [
            [
                channel_activations / channel_sum
                for channel_activations, channel_sum in zip(image_activations, image_sums)
            ]
            for image_activations, image_sums in zip(activations, sums)
        ]
        negative_probabilities = [
            [
                (1 - channel) / (len(channel) - channel.sum())
                for channel in image
            ]
            for image in positive_probabilities
        ]
        positives = [
            [
                [channel.positives[i] for i in np.random.choice(len(channel.positives), self.n_positives, replace=False, p=channel_probabilities)]
                for channel, channel_probabilities in zip(image, image_probabilities)
            ]
            for image, image_probabilities in zip(values, positive_probabilities)
        ]
        negatives = [
            [
                [channel.negatives[i] for i in np.random.choice(len(channel.negatives), self.n_negatives, replace=False, p=channel_probabilities)]
                for channel, channel_probabilities in zip(image, image_probabilities)
            ]
            for image, image_probabilities in zip(values, negative_probabilities)
        ]
        return positives, negatives
