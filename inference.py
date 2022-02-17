from argparse import ArgumentParser

import PIL
import torch
import torchvision.utils

import data
import models

PIL.Image.MAX_IMAGE_PIXELS = None

parser = ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("image", type=str, default="inference.tiff")
args = parser.parse_args()

model: models.Model = models.Model.load_from_checkpoint(args.model)

dataset = data.SamplesDataset(args.image, crop_size=2000)
image = dataset[0].unsqueeze(0)

maps = model.attention_network(image)
maps = maps[0].unsqueeze(1)

torchvision.utils.save_image(torch.vstack((image, maps)), 'inferred_attention_maps.png')
