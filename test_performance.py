import sys

import torch

import data
import models


dataset = data.MoNuSegValidationDataset(sys.argv[1])
d = list(dataset)
images = torch.stack([e[0] for e in d])
masks = torch.stack([e[1] for e in d])

model: models.Model = models.Model.load_from_checkpoint(sys.argv[2])

attention_maps = model.attention_network(images)
predicted_masks = attention_maps > 0.5

masks = masks > 0.5  # turn it into a boolean tensor
tp = torch.logical_and(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)
fp_fn = torch.logical_xor(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)

dice_scores = 2 * tp / (2 * tp + fp_fn)
dice_score = dice_scores.mean(dim=0)[sys.argv[3]]

print(dice_score)
