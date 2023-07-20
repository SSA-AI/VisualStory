import os
import matplotlib
import numpy as np
import torch
import torchvision
from huggingface_hub.utils import tqdm
from torchvision import models, transforms
import torchvision.transforms as transforms
import random
import clip
from PIL import Image
import os
import skimage
from abc import ABC, abstractmethod
from typing import Dict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import itertools

"""
In this script I implement the interaction with CLIP functions using CIFAR10 as a first stage before 
implementations using a different feature extractor.

This is the most recent implementation in this folder.

"""


class DataHandler(ABC):
    def get_sample(self):
        pass

    def visualize_sample(self, images_dict: Dict):
        pass


from torch.utils.data import random_split, DataLoader


class DataHandlerCifar10(DataHandler):
    def __init__(self, data_dir='./data', batch_size=64, val_split=0.2):
        self.data_dir = data_dir
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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


class DataHandlerSKImage(DataHandler):
    def __init__(self, preprocess=None):
        self.preprocess = preprocess

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


class VisualizeSimilarityMatrix:
    def __init__(self, clip_model, feature_extractor=None):
        self.model = clip_model
        self.extractor = feature_extractor

    def cifar10(self, images, texts, images_dict):
        # Building features
        image_input = torch.tensor(np.stack(images))
        text_tokens = clip.tokenize(["This is a " + desc for desc in texts])

        with torch.no_grad():
            if self.extractor is None:
                image_features = self.model.encode_image(image_input).float()
            else:
                image_features = self.extractor(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Calculating cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        count = len(images_dict)

        # Plotting the similarity matrix
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)

        # Plotting the captions as yticks
        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])

        # Plotting the tiny images above
        for i, image in enumerate(images):
            image = np.transpose(image.numpy(), (1, 2, 0))  # convert from Tensor image and transpose
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        # Writing the similarity values in the middle
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
        plt.show()

    def skimage(self, images, texts, descriptions, original_images):

        image_input = torch.tensor(np.stack(images))
        text_tokens = clip.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            if self.extractor is None:
                image_features = self.model.encode_image(image_input).float()
            else:
                image_features = self.extractor(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Calculating cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        count = len(descriptions)

        # ploting the similarity matrix
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        # plt.colorbar()

        # ploting the captions as yticks
        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])

        # ploting the tiny images above
        for i, image in enumerate(original_images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        # Writing the similarity values in the middle
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
        plt.show()


class ClipLoader:
    def __init__(self, model_name, print_info_flag=True):
        self.model_name = model_name
        self.print_info_flag = print_info_flag
    def load(self):
        # Loading the model
        clip.available_models()
        model, preprocess = clip.load(self.model_name)
        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        if self.print_info_flag:
            print("Torch version:", torch.__version__)
            print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
            print("Input resolution:", input_resolution)
            print("Context length:", context_length)
            print("Vocab size:", vocab_size)

        return model, preprocess


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, model_name, num_features):
        super(ResNet50FeatureExtractor, self).__init__()
        model = torchvision.models.__dict__[model_name](pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class LossObj:
    def __init__(self, loss_name='mse'):
        if loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError("Loss function not supported")

    def get_loss(self):
        return self.loss_func


class Trainer:
    def __init__(self, feature_extractor, clip_model, loss_obj, device):
        self.feature_extractor = feature_extractor
        self.clip_model = clip_model
        self.loss_obj = loss_obj
        self.device = device
        self.writer = SummaryWriter()  # TensorBoard writer

    def train(self, dataloader, classes, optimizer, num_epochs=10, subset_size=100):
        self.feature_extractor.train()
        for epoch in tqdm(range(num_epochs)):
            # Create an iterator from the dataloader and use itertools.islice to only get a subset of it
            data_iter = iter(dataloader)
            for i, (images, labels) in enumerate(itertools.islice(data_iter, subset_size)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Convert numerical labels to text labels
                text_labels = [classes[label] for label in labels]

                # Generate image features using feature extractor
                image_features = self.feature_extractor(images)

                # Generate label features using CLIP
                label_features = self.clip_model.encode_text(clip.tokenize(text_labels))

                # Compute loss
                loss = self.loss_obj.get_loss()(image_features, label_features)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{subset_size}], Loss: {loss.item():.4f}')

                # Log the loss to TensorBoard
                self.writer.add_scalar('Training Loss', loss.item(), epoch * subset_size + i)

        # Close the TensorBoard writer
        self.writer.close()


def train_fc_layer(feature_extractor, clip_model, loss_name, device, dataloader, lr=0.001, num_epochs=10):
    # Initialize loss object
    loss_obj = LossObj(loss_name)

    # Initialize trainer
    trainer = Trainer(feature_extractor, clip_model, loss_obj, device)

    # Define optimizer
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr)

    # Train
    trainer.train(dataloader, optimizer, num_epochs=num_epochs)


###################################
def main_feature_extractor_skimage(feature_extractor):
    # ** get model name from .yaml
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # ** obtain from .yaml file
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer"
    }
    data_handler = DataHandlerSKImage(preprocess=preprocess)
    images, texts, original_images, filename_list = data_handler.get_sample(descriptions=descriptions)

    # visualize text-image similarity:
    VisualizeSimilarityMatrix(clip_model=model, feature_extractor=feature_extractor)\
        .skimage(images=images, texts=texts,
                 descriptions=descriptions,
                 original_images=original_images)


def main_feature_extractor_cifar10(feature_extractor):
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # Create DataHandler instance and download data
    image_dir = "/Users/shahafasban/PycharmProjects/Data/cifar10_test"
    data_handler = DataHandlerCifar10(data_dir=image_dir)
    data_handler.download_and_save()

    # Get sample images
    images_dict = data_handler.get_sample()

    # preprocess data // cifar10
    images, texts, original_images = PreProcessSet(clip_preprocess=preprocess).preprocess_cifar10(
        sample_dict=images_dict)

    # visualize similarity matrix for the cifar10 set
    #################################################
    VisualizeSimilarityMatrix(clip_model=model, feature_extractor=feature_extractor)\
        .cifar10(images=images, texts=texts, images_dict=images_dict)


def main_skimage():

    # ** get model name from .yaml
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # ** obtain from .yaml file
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer"
    }
    data_handler = DataHandlerSKImage(preprocess=preprocess)
    images, texts, original_images, filename_list = data_handler.get_sample(descriptions=descriptions)

    ## visualize sample:
    # data_handler.visualize_sample(original_images, texts, filename_list)

    # visualize text-image similarity:
    VisualizeSimilarityMatrix(clip_model=model).skimage(images=images, texts=texts,
                                                        descriptions=descriptions,
                                                        original_images=original_images)


def main_cifar10():
    # Loading the model
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # Create DataHandler instance and download data
    image_dir = "/Users/shahafasban/PycharmProjects/Data/cifar10_test"
    data_handler = DataHandlerCifar10(data_dir=image_dir)
    data_handler.download_and_save()

    # Get sample images and visualize
    images_dict = data_handler.get_sample()
    # data_handler.visualize_sample(images_dict)

    # Separate the dictionary into two lists // debug step
    labels = list(images_dict.keys())
    image_list = list(images_dict.values())

    # preprocess data // cifar10
    images, texts, original_images = PreProcessSet(clip_preprocess=preprocess).preprocess_cifar10(sample_dict=images_dict)

    # visualize similarity matrix for the cifar10 set
    #################################################
    VisualizeSimilarityMatrix(clip_model=model).cifar10(images=images, texts=texts, images_dict=images_dict)


if __name__ == "__main__":

    # main_cifar10()
    # print("Done comparing CIFAR10")

    # main_skimage()
    # print("Done comparing skimage")

    ############## Train Main ###########
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize feature extractor (ResNet50FeatureExtractor from previous example)
    model_name_str = 'resnet50'
    print(f"Loading feature extractor: {model_name_str}...")
    feature_extractor = ResNet50FeatureExtractor(model_name_str, 512).to(device)

    # Initialize ClipLoader
    clip_model_str = "ViT-B/32"
    print(f"Loading CLIP model: {clip_model_str}...")
    clip_loader, preprocess = ClipLoader(clip_model_str).load()

    ########## Test naive performance ########
    # main_feature_extractor_skimage(feature_extractor=feature_extractor)
    # main_feature_extractor_cifar10(feature_extractor)
    # print("Done")
    ########## continue to train ###########

    # Create DataHandler instance and download data
    data_dir = "/Users/shahafasban/PycharmProjects/Data/cifar10_test"
    data_handler = DataHandlerCifar10(data_dir=data_dir, batch_size=64, val_split=0.2)
    data_handler.download_and_save()

    # Split the dataset into training and validation sets
    train_loader, val_loader, classes = data_handler.split_train_val()

    # Initialize loss object
    loss_obj = LossObj('mse')

    # Initialize trainer
    trainer = Trainer(feature_extractor, clip_loader, loss_obj, device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.001)

    # Train
    trainer.train(train_loader, classes, optimizer, num_epochs=2)

    print("Done :-)")
