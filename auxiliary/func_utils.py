import yaml
import torch
import argparse
from entities.trainer import LossObj, Trainer
from entities.visualization import VisualizeSimilarityMatrix
from entities.data_handlers import DataHandlerSKImage, DataHandlerCifar10, PreProcessSet
from entities.model_loaders import ClipLoader, LoadTrainedModel, ResNetXFeatureExtractor


def main_skimage():

    # ** get model name from .yaml
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # get descriptions
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    descriptions = config["descriptions"]

    data_handler = DataHandlerSKImage(preprocess=preprocess)
    images, texts, original_images, filename_list = data_handler.get_sample(descriptions=descriptions)

    ## visualize sample:
    data_handler.visualize_sample(original_images, texts, filename_list)

    # visualize text-image similarity:
    VisualizeSimilarityMatrix(clip_model=model).skimage(images=images, texts=texts,
                                                        descriptions=descriptions,
                                                        original_images=original_images)


def main_cifar10(feature_extractor=None):
    # Loading the model
    model, preprocess = ClipLoader(model_name="ViT-B/32").load()

    # Create DataHandler instance and download data
    image_dir = "/Users/shahafasban/PycharmProjects/Data/cifar10_test"
    data_handler = DataHandlerCifar10(data_dir=image_dir)
    data_handler.download_and_save()

    # Get sample images and visualize
    images_dict = data_handler.get_sample()
    # data_handler.visualize_sample(images_dict)

    # preprocess data // cifar10
    images, texts, original_images = PreProcessSet(clip_preprocess=preprocess).preprocess_cifar10(sample_dict=images_dict)

    # visualize similarity matrix for the cifar10 set
    VisualizeSimilarityMatrix(clip_model=model, feature_extractor=feature_extractor).cifar10(images=images,
                                                                                             texts=texts,
                                                                                             images_dict=images_dict)


def parse_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    return parser.parse_args()


def load_trained_feature_extractor_main(model_name, num_features, epoch_num=None, weights_dir=None):

    if weights_dir is None:
        raise ValueError("Trained model directory cannot be found. Please add a valid directory path.")
    else:
        # Define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize feature extractor (ResNet50FeatureExtractor from previous example)
        feature_extractor = ResNetXFeatureExtractor(model_name, num_features).to(device)

        # Initialize LoadTrainedModel
        loader = LoadTrainedModel(feature_extractor, weights_dir)

        # Load the latest model weights
        loader.load_weights(epoch_num=epoch_num)

        return feature_extractor


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
    images, texts, original_images = \
        PreProcessSet(clip_preprocess=preprocess).preprocess_cifar10(sample_dict=images_dict)

    # visualize similarity matrix for the cifar10 set
    VisualizeSimilarityMatrix(clip_model=model, feature_extractor=feature_extractor)\
        .cifar10(images=images, texts=texts, images_dict=images_dict)


def train_fc_layer(feature_extractor, clip_model, loss_name, device, dataloader, lr=0.001, num_epochs=10):
    # Initialize loss object
    loss_obj = LossObj(loss_name)

    # Initialize trainer
    trainer = Trainer(feature_extractor, clip_model, loss_obj, device)

    # Define optimizer
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr)

    # Train
    trainer.train(dataloader, optimizer, num_epochs=num_epochs)
