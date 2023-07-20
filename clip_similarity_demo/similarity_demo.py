import os
import numpy as np
import torch
from PIL import Image
import clip
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Determine if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############## Image Download/load/save ######################
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CIFAR10DataLoader:
    def __init__(self, root_dir='./data', batch_size=4, num_workers=2):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self._download_and_load()

    def _download_and_load(self):
        self.trainset = torchvision.datasets.CIFAR10(root=self.root_dir, train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=self.num_workers)

    def get_data(self):
        return next(iter(self.trainloader))


class CIFAR10ImageSaver:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def save_images(self, images, filenames):
        for img, filename in zip(images, filenames):
            self._imsave(img, filename)

    @staticmethod
    def _imsave(img, filename):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(filename)



# Usage
test_image_dir = "/Users/shahafasban/PycharmProjects/Data/cifar10_test"
data_loader = CIFAR10DataLoader(test_image_dir)
inputs, classes = data_loader.get_data()

image_saver = CIFAR10ImageSaver()

for i in range(10):
    image_saver.imsave(inputs[i], f'cifar10_image_{i}.png')

####################################

class ClipModel:
    def __init__(self, model_name="ViT-B/32"):
        self.model, self.preprocess = clip.load(model_name)
        self.model = self.model.to(device).eval()

    def tokenize(self, text):
        return clip.tokenize(text).to(device)

    def encode_image(self, image_input):
        with torch.no_grad():
            return self.model.encode_image(image_input).float()

    def encode_text(self, text_tokens):
        with torch.no_grad():
            return self.model.encode_text(text_tokens).float()

class ImageProcessor:
    def __init__(self, model_preprocess):
        self.preprocess = model_preprocess

    def process(self, image):
        return self.preprocess(image)

class ImageLoader:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def load(self, filename):
        image = Image.open(filename).convert("RGB")
        return self.image_processor.process(image)

class SimilarityCalculator:
    def calculate(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy() @ image_features.cpu().numpy().T

class ClassificationProcessor:
    def __init__(self, clip_model):
        self.clip_model = clip_model

    def process(self, image_features, descriptions):
        text_tokens = self.clip_model.tokenize(descriptions)
        text_features = self.clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs.cpu().topk(5, dim=-1)

class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

# Using the above classes
clip_model = ClipModel()
image_processor = ImageProcessor(clip_model.preprocess)
image_loader = ImageLoader(image_processor)

# Provide a list of files and descriptions
files = ["file1.png", "file2.jpg"]
descriptions = ["This is a cat.", "This is a dog."]

images = []
for filename in files:
    image = image_loader.load(filename)
    images.append(image)

image_input = torch.tensor(np.stack(images)).to(device)
text_tokens = clip_model.tokenize(descriptions)

image_features = clip_model.encode_image(image_input)
text_features = clip_model.encode_text(text_tokens)

similarity_calculator = SimilarityCalculator()
similarity = similarity_calculator.calculate(image_features, text_features)

# For image classification
classification_processor = ClassificationProcessor(clip_model)
transforms = Compose([Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_dataset = CustomCIFAR10(os.path.expanduser("~/.cache"), train=True, transform=transforms, download=True)

text_descriptions = [f"This is a photo of a {label}" for label in cifar10_dataset.classes]

dataloader = DataLoader(cifar10_dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in dataloader:
    images = images.to(device)
    image_features = clip_model.encode_image(images)
    top_probs, top_labels = classification_processor.process(image_features, text_descriptions)
