# Data loader/saver/captions

import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10Data:
    def __init__(self, root='./data'):
        self.root = root
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Define a transform to normalize the data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def download_and_load(self):
        # Download and load the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.transform)
        return trainset, testset


class CIFAR10Captions:
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes

    def create_captions(self):
        captions = []
        for image, label in self.dataset:
            captions.append(self.classes[label])
        return captions


# Usage:

cifar10_data = CIFAR10Data()
trainset, testset = cifar10_data.download_and_load()

train_captions_creator = CIFAR10Captions(trainset, cifar10_data.classes)
train_captions = train_captions_creator.create_captions()

test_captions_creator = CIFAR10Captions(testset, cifar10_data.classes)
test_captions = test_captions_creator.create_captions()
