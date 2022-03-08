import glob
import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop

import skimage.draw
import xml.etree.ElementTree as ET
import numpy as np


class SamplesDataset(Dataset):
    def __init__(self, image_path: str = "image.tiff", crop_size: int = 500, epsilon: float = 0.05) -> None:
        image = Image.open(image_path)
        image = ToTensor()(image)[:3]  # Discard the alpha band
        image = Grayscale(1)(image)
        self.image = image
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon

    def __len__(self):
        return 250

    def __getitem__(self, _) -> Tensor:
        image = self.cropper(self.image)
        return (1 - self.epsilon) * image + self.epsilon


class MultiSamplesDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 500, epsilon: float = 0.05) -> None:
        image_files = glob.glob(os.path.join(image_directory, '*.tif'))
        images = map(Image.open, image_files)
        images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon
    
    def __len__(self):
        return 250

    def __getitem__(self, _) -> Tensor:
        idx = random.randrange(0, len(self.images))
        image = self.images[idx]
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


class MoNuSegDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05):
        image_files = glob.glob(os.path.join(image_directory, '*.tif'))
        images = map(Image.open, image_files)
        images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


class MoNuSegValidationDataset(Dataset):
    def __init__(self, directory, epsilon: float = 0.05):
        image_files = glob.glob(os.path.join(directory, '*tif'))
        mask_files = [
            f'{os.path.join(directory, os.path.splitext(os.path.basename(image_file))[0])}.xml'
            for image_file in image_files
        ]

        images = map(Image.open, image_files)
        images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(self.binary_mask_from_xml_file, mask_files)
        masks = map(torch.Tensor, masks)
        self.masks = list(masks)

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask.unsqueeze(0)

    @classmethod
    def binary_mask_from_xml_file(cls, xml_file_path, image_shape=(1000, 1000)):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        def vertex_element_to_tuple(vertex_element):
            col = float(vertex_element.get('X'))
            row = float(vertex_element.get('Y'))
            return round(row), round(col)

        mask = np.zeros(image_shape, dtype=np.uint8)
        for region in root.iter('Region'):
            vertices = map(vertex_element_to_tuple, region.iter('Vertex'))
            rows, cols = np.array(list(zip(*vertices)))

            rr, cc = skimage.draw.polygon(rows, cols, mask.shape)
            mask[rr, cc] = 1

        return mask


def plotable(image):
    return torch.moveaxis(image, 0, -1)


if __name__ == "__main__":
    dataset = SamplesDataset("image.tiff")

    plt.imshow(plotable(dataset.image))
    plt.show()

    fig, axs = plt.subplots(5, 5)
    for row in axs:
        for cell in row:
            cell.imshow(plotable(dataset[0]))
    fig.tight_layout()
    plt.show()
