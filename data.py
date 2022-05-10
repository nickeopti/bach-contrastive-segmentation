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

import scipy.io
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
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(image_directory, '*.tif'))
        images = map(Image.open, image_files)
        if grey_scale:
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


class MoNuSegWSIDataset(Dataset):
    def __init__(self, directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False) -> None:
        image_files = glob.glob(os.path.join(directory, '*', '*.svs'))

        # slide = openslide.OpenSlide(image_files[0])
        # slide.dimensions
        # r = slide.read_region((100000,80000), 0, (10, 10))
        # t = ToTensor()(r)



class MoNuSegValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(directory, '*tif'))
        mask_files = [
            f'{os.path.join(directory, os.path.splitext(os.path.basename(image_file))[0])}.xml'
            for image_file in image_files
        ]

        images = map(Image.open, image_files)
        if grey_scale:
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


class GlaSDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05):
        image_files = [f for f in glob.glob(os.path.join(image_directory, '*.bmp')) if 'anno' not in f]
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


class GlaSValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05):
        files = glob.glob(os.path.join(directory, '*.bmp'))
        image_files = [f for f in files if 'anno' not in f]
        mask_files = [f for f in files if 'anno' in f]

        images = map(Image.open, image_files)
        images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(Image.open, mask_files)
        masks = map(Grayscale(1), images)
        masks = map(torch.Tensor, masks)
        self.masks = [(m > 0).type(torch.float) for m in masks]

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask


class CoNSePDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(image_directory, 'Images', '*.png'))
        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


class CoNSePValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(directory, 'Images', '*.png'))
        mask_files = [
            f'{os.path.join(directory, "Labels", os.path.splitext(os.path.basename(image_file))[0])}.mat'
            for image_file in image_files
        ]

        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(scipy.io.loadmat, mask_files)
        masks = [mask['inst_map'] > 0 for mask in masks]
        masks = map(torch.Tensor, masks)
        self.masks = list(masks)

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask.unsqueeze(0)


class TNBCDataset(Dataset):
    def __init__(self, image_directory: str, crop_size: int = 250, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(image_directory, 'Slide_*', '*.png'))
        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)
        self.cropper = RandomCrop(crop_size)
        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        image = self.cropper(image)
        return (1 - self.epsilon) * image + self.epsilon


class TNBCValidationDataset(Dataset):
    def __init__(self, directory: str, epsilon: float = 0.05, grey_scale: bool = False):
        image_files = glob.glob(os.path.join(directory, 'Slide_*', '*.png'))
        mask_files = [image_file.replace('Slide', 'GT') for image_file in image_files]

        images = map(Image.open, image_files)
        if grey_scale:
            images = map(Grayscale(1), images)
        images = map(ToTensor(), images)
        self.images = list(images)

        masks = map(Image.open, mask_files)
        masks = map(Grayscale(1), masks)
        masks = map(ToTensor(), masks)
        masks = [(mask > 0).type(torch.float) for mask in masks]
        self.masks = list(masks)

        self.epsilon = epsilon

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][:3]  # Remove alpha band
        mask = self.masks[idx]
        return (1 - self.epsilon) * image + self.epsilon, mask


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
