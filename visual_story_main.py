from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from blocks.feature_extractor import FeatureExtractor
from blocks.semantic_segmentation import SemanticSegmentation
from blocks.transformer import TransformerModel
from entities.download_cifar10 import download_cifar10_dataset
from entities.visualization import VisualizeStory
import ssl


def main():

    # Download set if required
    data_dir = "/Users/shahafasban/PycharmProjects/Data"
    ssl._create_default_https_context = ssl._create_unverified_context
    download_cifar10_dataset(data_dir)

    # Path to CIFAR-10 data
    cifar10_train_data_path = data_dir + '/cifar-10-batches-py/data_batch_1'
    cifar10_test_data_path = data_dir + '/cifar-10-batches-py/test_batch'

    # Initialize dataset
    train_data = CIFAR10(root=cifar10_train_data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = CIFAR10(root=cifar10_test_data_path, train=False, download=True, transform=transforms.ToTensor())

    # Create a DataLoader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Initialize segmentation model
    segmentation_model = SemanticSegmentation()

    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    # Initialize transformer model
    transformer_model = TransformerModel()

    for loaders in [train_loader, test_loader]:
        for images, labels in loaders:
            # Segment images
            segmented_images = segmentation_model.segment(images)

            # Extract features
            image_features = feature_extractor.extract_features(segmented_images)

            # Generate text
            generated_text = transformer_model.generate_text(image_features)

            # Print or save generated text
            visualizer = VisualizeStory(images, generated_text)
            visualizer.display()


if __name__ == "__main__":
    main()
