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
