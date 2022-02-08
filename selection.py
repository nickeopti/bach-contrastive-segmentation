from abc import abstractmethod
import torch
import torch.nn as nn


class ContrastiveSelector:
    @abstractmethod
    def select(self, attended_image):
        pass


class ContrastiveRegionsSelector(ContrastiveSelector):
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
    
    def transform_where_result_to_indices(self, channel_index, where_indices):
        _, _, row, col = where_indices
        return list(
            zip(
                [channel_index] * row.numel(),
                (row * self.stride).tolist(),
                (col * self.stride).tolist(),
                [self.size] * row.numel(),
            )
        )


class ContrastiveRegionsQuantileSelector(ContrastiveRegionsSelector):
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

        positives = self.transform_where_result_to_indices(0, torch.where(slide_sum > self.size**2 / 2))
        # _, _, rows, cols = torch.where(slide_sum > self.size**2 / 2)
        # positives = list(zip((rows*self.stride).tolist(), (cols*self.stride).tolist(), [self.size]*rows.numel()))

        negatives = self.transform_where_result_to_indices(0, torch.where(slide_sum < self.size**2 / 2))
        # _, _, rows, cols = torch.where(slide_sum < self.size**2 / 2)
        # negatives = list(zip((rows*self.stride).tolist(), (cols*self.stride).tolist(), [self.size]*rows.numel()))

        return positives, negatives


class MultiChannelQuantileSelector(ContrastiveRegionsQuantileSelector):
    def __init__(self, size: int = 50, stride: int = 5, quantile: float = 0.9, device=None) -> None:
        super().__init__(size, stride, quantile, device)

    def select(self, attended_image):
        return [
            self.transform_where_result_to_indices(
                i,
                torch.where(
                    self.f(
                        (channel > torch.quantile(channel, self.quantile))
                        .type(torch.float)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    > self.size ** 2 / 2
                ),
            )
            for i, channel in enumerate(attended_image[0])  # Assuming batch size of 1
        ]
