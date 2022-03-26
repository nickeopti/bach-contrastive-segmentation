import csv
import os
import sys

import torch

import data
import models


dataset = data.MoNuSegValidationDataset(sys.argv[1])
d = list(dataset)
images = torch.stack([e[0] for e in d])
masks = torch.stack([e[1] for e in d]) > 0.5

def test_performance(version):
    with open(f'logs/default/version_{version}/metrics.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        best_epoch = max(reader, key=lambda e: e['best_val_mdc'])
        epoch = best_epoch['epoch']
        step = best_epoch['step']
        n_classes = len([k for k in best_epoch.keys() if k.startswith('mdc_c')])
        channel = max(range(n_classes), key=lambda i: best_epoch[f'mdc_c{i+1}'])

    model: models.Model = models.Model.load_from_checkpoint(f'logs/default/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt')

    attention_maps = model.attention_network(images)
    predicted_masks = attention_maps > 0.5

    tp = torch.logical_and(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)
    fp_fn = torch.logical_xor(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)

    dice_scores = 2 * tp / (2 * tp + fp_fn)
    dice_score = dice_scores.mean(dim=0)[channel].item()

    return dice_score, best_epoch['best_val_mdc']


if __name__ == '__main__':
    with open(sys.argv[2]) as info_file:
        reader = csv.reader(info_file)
        info = [(version, group) for version, group in reader if os.path.exists(f'logs/default/version_{version}')]

    results = {i[1]: [] for i in info}
    for version, group in info:
        print(version, group)

        test, val = test_performance(version)
        results[group].append((test, val))

        print('DICE score:', test)
        print()

    for group, res in results.items():
        score = max(res, key=lambda e: e[1])[0]
        print(group, score)
