import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop


class SamplesDataset(Dataset):
    def __init__(self, image_path, crop_size=500, epsilon=0.05) -> None:
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


def plotable(image):
    return torch.moveaxis(image, 0, -1)


if __name__ == '__main__':
    dataset = SamplesDataset('image.tiff')

    plt.imshow(plotable(dataset.image))
    plt.show()

    fig, axs = plt.subplots(5, 5)
    for row in axs:
        for cell in row:
            cell.imshow(plotable(dataset[0]))
    fig.tight_layout()
    plt.show()
