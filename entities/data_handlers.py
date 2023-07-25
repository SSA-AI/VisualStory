import os
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np
import skimage
import torchvision
from typing import Dict
import random
from abc import ABC
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import skimage


class DataHandler(ABC):
    def get_sample(self):
        pass

    def visualize_sample(self, images_dict: Dict):
        pass


class DataHandlerCifar10(DataHandler):
    def __init__(self, data_dir='./data', batch_size=64, val_split=0.2):
        self.data_dir = data_dir
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                         (0.2023, 0.1994, 0.2010))])
        self.trainset = None
        self.batch_size = batch_size
        self.val_split = val_split

    def download_and_save(self):
        if os.path.exists(os.path.join(self.data_dir, 'cifar-10-batches-py')):
            print("Data exists in target folder.")
        self.trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True,
                                                     transform=self.transform)

    def get_sample(self, descriptions=None):
        if self.trainset is None:
            self.trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=False, transform=self.transform)
        images_dict = {}
        counts = {i: 0 for i in range(10)}
        for img, label in self.trainset:
            if label not in images_dict or random.random() < 1.0 / (counts[label] + 1):
                images_dict[label] = img
            counts[label] += 1
        images_dict = {self.classes[i]: img for i, img in images_dict.items()}
        return images_dict

    def visualize_sample(self, images_dict):
        # Convert dictionary to two lists
        labels = list(images_dict.keys())
        images = list(images_dict.values())

        # Plot images in a gallery
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.ravel()

        for i in range(10):
            img = images[i]
            img = img / 2 + 0.5  # unnormalize
            img = np.transpose(img, (1, 2, 0))  # convert from Tensor image
            axs[i].imshow(img)
            axs[i].set_title(labels[i], fontsize=16)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    def split_train_val(self):
        if self.trainset is None:
            self.download_and_save()

        # Determine the lengths of splits
        val_len = int(self.val_split * len(self.trainset))
        train_len = len(self.trainset) - val_len

        # Split the dataset
        train_set, val_set = random_split(self.trainset, [train_len, val_len])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, self.classes


class PreProcessSet:
    def __init__(self, clip_preprocess):
        self.preprocess = clip_preprocess

    def preprocess_cifar10(self, sample_dict):
        # Convert images to PIL format and preprocess
        original_images = []
        images = []
        texts = []

        for label, image in sample_dict.items():
            original_images.append(image)
            image = Image.fromarray((np.transpose((image.numpy() * 0.5 + 0.5), (1, 2, 0)) * 255).astype(np.uint8))
            images.append(self.preprocess(image))
            texts.append(label)

        return images, texts, original_images

    def preprocess_skimage(self, images, texts):
        pass


class ImageNetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_data(self):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        return dataset


class SKImageDataset(Dataset):
    def __init__(self, descriptions, preprocess=None):
        self.descriptions = descriptions
        self.preprocess = preprocess
        self.filenames = [filename for filename in os.listdir(skimage.data_dir) if
                          filename.endswith(".png") or filename.endswith(".jpg")]
        self.filenames = [filename for filename in self.filenames if os.path.splitext(filename)[0] in descriptions]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        name = os.path.splitext(filename)[0]
        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
        if self.preprocess is not None:
            image = self.preprocess(image)
        text = self.descriptions[name]
        return image, text


class DataHandlerSKImage(DataHandler):
    def __init__(self, preprocess=None, batch_size=2, num_workers=1):
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataloader(self, descriptions=None):
        dataset = SKImageDataset(descriptions, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dataloader

    def get_sample(self, descriptions=None):
        original_images = []
        images = []
        texts = []
        filename_list = []
        for filename in [filename for filename in os.listdir(skimage.data_dir) if
                         filename.endswith(".png") or filename.endswith(".jpg")]:
            name = os.path.splitext(filename)[0]
            if name not in descriptions:
                continue

            image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
            original_images.append(image)
            images.append(self.preprocess(image))
            texts.append(descriptions[name])
            filename_list.append(filename)
        return images, texts, original_images, filename_list

    def visualize_sample(self, images, texts, filenames):

        plt.figure(figsize=(16, 5))
        for idx, (image, filename) in enumerate(zip(images, filenames)):
            plt.subplot(2, 4, idx + 1)
            plt.imshow(image)
            plt.title(f"{filename}\n{texts[idx]}")
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()


