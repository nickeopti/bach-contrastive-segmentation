from argparse import ArgumentParser
import math

import PIL
import torch
import torchvision.utils

import data
import models

PIL.Image.MAX_IMAGE_PIXELS = None

parser = ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("image", type=str, default="inference.tiff")
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--offset", type=float, default=0)
args = parser.parse_args()

model: models.Model = models.Model.load_from_checkpoint(args.model)

dataset = data.SamplesDataset(args.image, crop_size=4000)
image = dataset[0].unsqueeze(0) * args.scale + args.offset

maps = model.attention_network(image)
maps = maps[0].unsqueeze(1)

nrow = math.ceil(math.sqrt(len(maps) + 1))  # Number of images per row; ie number of columns
torchvision.utils.save_image(
    torch.vstack((image, maps)),
    f'inferred_attention_maps_{args.model.split("/")[2]}_scale_{str(args.scale).replace(".", "_")}_offset_{str(args.offset).replace(".", "_")}.png',
    nrow=nrow
)
