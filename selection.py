import torch
import torch.nn as nn


def contrastive_areas(attended_image, size, stride=5):
    f = nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=size, stride=stride, bias=False
    )
    f.weight = nn.Parameter(
        torch.ones((1, 1, size, size), dtype=torch.float, device=attended_image.device),
        requires_grad=False,
    )

    def positive_indices(channel_index, where_indices):
        _, _, row, col = where_indices
        return list(
            zip(
                [channel_index] * row.numel(),
                (row * stride).tolist(),
                (col * stride).tolist(),
                [size] * row.numel(),
            )
        )

    return [
        positive_indices(
            i,
            torch.where(
                f(
                    (channel > torch.quantile(channel, 0.9))
                    .type(torch.float)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                > size ** 2 / 2
            ),
        )
        for i, channel in enumerate(attended_image[0])  # Assuming batch size of 1
    ]
