import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop

import data
import plot

dataset = data.MoNuSegValidationDataset(sys.argv[1])
d = list(dataset)
images = torch.stack([e[0] for e in d])
masks = torch.stack([e[1] for e in d])

masks = masks > 0.5  # turn it into a boolean tensor

results = []
for t in np.linspace(0, 1, 100):
    tp = torch.logical_and(masks, images < t).flatten(start_dim=2).sum(dim=2)
    fp_fn = torch.logical_xor(masks, images < t).flatten(start_dim=2).sum(dim=2)

    dice_scores = 2 * tp / (2 * tp + fp_fn)
    results.append((t, dice_scores.mean().item()))


m = max(results, key=lambda v: v[1])
print(m)

plt.plot(*zip(*results))
plt.savefig("threshold_baseline.png")


dataset = data.MoNuSegValidationDataset(sys.argv[2])
d = list(dataset)
images = torch.stack([e[0] for e in d])
masks = torch.stack([e[1] for e in d])


tp = torch.logical_and(masks, images < m[0]).flatten(start_dim=2).sum(dim=2)
fp_fn = torch.logical_xor(masks, images < m[0]).flatten(start_dim=2).sum(dim=2)

dice_scores = 2 * tp / (2 * tp + fp_fn)
print(dice_scores.mean().item())


plot.plot_mask(images, masks, (images < m[0]).type(torch.float), path="threshold.png")
