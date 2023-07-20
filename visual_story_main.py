import ssl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, ToTensor, Compose

from blocks.ClipCaptionProcessor import CLIPProcessor
from blocks.transformer import TransformerModel
from entities.visualization import VisualizeStory
from blocks.feature_extractor import FeatureExtractor
from blocks.semantic_segmentation import SemanticSegmentation


#########################################
import clip
import torch


class CLIPProcessor:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def process_images(self, images):
        return self.model.encode_image(images)

    def generate_text(self, image_features):
        text = clip.tokenize(["A photo of a {label}."])
        text = text.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

#########################################
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from typing import List


class VisualizeStory:
    def __init__(self, images, texts):
        """
        Args:
            images (list): List of images.
            texts (list): List of corresponding texts.
        """
        self.images = images
        self.texts = texts

    def display(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for img, txt in zip(self.images, self.texts):
            ax.imshow(img)
            ax.title.set_text(txt)
            ax.axis('off')

            display(fig)

            # Wait for user input and then clear the output
            input("Press enter to continue...")
            clear_output(wait=True)

#########################################

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TransformerModel:
    def __init__(self, device="cuda"):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate_text(self, inputs):
        """
        Args:
            inputs (torch.Tensor): The input features of shape (B, F)
        Returns:
            generated_text (list of str): The generated text for each input
        """
        generated_text = []
        with torch.no_grad():
            inputs = inputs.to(self.device)
            for input in inputs:
                input = input.unsqueeze(0)
                generated = self.model.generate(input, max_length=150, temperature=0.7, num_return_sequences=1)
                text = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
                generated_text.append(text[0])
        return generated_text

#########################################

def main():
    # Download set if required
    data_dir = "/Users/shahafasban/PycharmProjects/Data"
    ssl._create_default_https_context = ssl._create_unverified_context

    # Path to CIFAR-10 data
    cifar10_train_data_path = data_dir + '/cifar-10-batches-py/data_batch_1'
    cifar10_test_data_path = data_dir + '/cifar-10-batches-py/test_batch'

    # Initialize dataset
    train_data = CIFAR10(root=cifar10_train_data_path, train=True, download=True, transform=Compose([Resize((224, 224)), ToTensor()]))
    test_data = CIFAR10(root=cifar10_test_data_path, train=True, download=True, transform=Compose([Resize((224, 224)), ToTensor()]))

    # Create a DataLoader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize CLIP Processor
    clip_processor = CLIPProcessor(device)

    for loaders in [train_loader, test_loader]:
        for images, labels in loaders:
            # Process images
            image_features = clip_processor.process_images(images.to(device))

            # Generate text
            generated_text = clip_processor.generate_text(image_features)

            # Print or save generated text
            visualizer = VisualizeStory(images, generated_text)
            visualizer.display()

if __name__ == "__main__":
    main()
