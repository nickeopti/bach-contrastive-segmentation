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


class Sampler:
    def __init__(self, n_positives: int = 10, n_negatives: int = 10) -> None:
        self.n_positives = n_positives
        self.n_negatives = n_negatives

    @abstractmethod
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        pass


class UniformSampler(Sampler):
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        return torch.randint(low=0, high=values.shape[-1], size=(values.shape[0], 2, values.shape[1], self.n_positives))


class ProbabilisticSampler(Sampler):
    def __init__(self, n_positives: int = 10, n_negatives: int = 10, alpha: float = 1, beta: float = 0.5) -> None:
        super().__init__(n_positives, n_negatives)
        self.alpha = alpha
        self.beta = beta

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        maxes = values.max(dim=-1, keepdim=True).values
        normalised_values = values / maxes
        
        exponentiated_values = normalised_values ** self.alpha
        # exponentiated_values = torch.sigmoid((normalised_values + self.beta) / self.alpha)

        sums = exponentiated_values.sum(dim=-1, keepdim=True)
        probabilities_positives = exponentiated_values / sums
        probabilities_negatives = (1 - probabilities_positives) / (probabilities_positives.shape[-1] - 1)

        indices_positives = torch.stack(
            [
                torch.multinomial(image, self.n_positives, replacement=True)
                for image in probabilities_positives
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.multinomial(image, self.n_negatives, replacement=True)
                for image in probabilities_negatives
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class TopKSampler(Sampler):
    def __init__(self, n_positives: int = 10, n_negatives: int = 10, k: int = 50) -> None:
        super().__init__(n_positives, n_negatives)
        self.k = k
    
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        top_k_positive_indices = torch.topk(values, k=self.k, dim=-1).indices
        top_k_negative_indices = torch.topk(-values, k=values.shape[-1] - self.k, dim=-1).indices

        indices_positives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n_positives,))]
                        for channel in image
                    ]
                )
                for image in top_k_positive_indices
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n_negatives,))]
                        for channel in image
                    ]
                )
                for image in top_k_negative_indices
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)
