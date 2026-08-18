from os.path import split
import random
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
class MSTAR(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.image_paths = []

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(dirpath, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        images= self.transform(image)
        return images
class GaussianNoise(object):
    """Gaussian Noise Augmentation for tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        noise = torch.randn(1,img.shape[1],img.shape[2])
        noise = torch.cat([noise,noise,noise],dim=0)
        return torch.clamp(img + self.sigma * noise,0.0,1.0)
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.Grayscale(3),
                                                transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([transforms.RandomRotation(30)], p=0.2),
                                                transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4)],p=0.4),
                                                transforms.ToTensor(),
                                                transforms.RandomApply([GaussianNoise()], p=0.2),
                                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                                         std=[0.5,0.5,0.5])])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'MSTAR': lambda: MSTAR(self.root_folder,transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(64),
                                                              n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
