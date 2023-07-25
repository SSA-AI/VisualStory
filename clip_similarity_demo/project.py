import argparse
import os
import matplotlib
import numpy as np
import torch
import torchvision
import yaml
from huggingface_hub.utils import tqdm
from torchvision import models, transforms
import torchvision.transforms as transforms
import clip
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from entities.data_handlers import DataHandlerSKImage, DataHandlerCifar10, PreProcessSet
from entities.model_loaders import ClipLoader, ResNetXFeatureExtractor, LoadTrainedModel
from entities.trainer import train_main
from entities.visualization import VisualizeSimilarityMatrix

"""
In this script I implement the interaction with CLIP functions using CIFAR10 as a first stage before 
implementations using a different feature extractor.

This is the most recent implementation in this folder.
"""


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

    # Separate the dictionary into two lists // debug step
    labels = list(images_dict.keys())
    image_list = list(images_dict.values())

    # preprocess data // cifar10
    images, texts, original_images = PreProcessSet(clip_preprocess=preprocess).preprocess_cifar10(sample_dict=images_dict)

    # visualize similarity matrix for the cifar10 set
    #################################################
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

    if weights_dir == None:
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


if __name__ == "__main__":

    args = get_args()
    config = parse_config(args.config)
    weights_dir = os.path.join(config["results_dir"], "training_results")
    num_features = config["num_features"]
    model_name = config["model_name_str"]
    read_epoch_num = config["read_epoch_num"]

    main_cifar10()
    print("Done comparing CIFAR10")

    # main_skimage()
    # print("Done comparing skimage")

    # Run trainng protocol
    # train_main()

    # loading trained model:

    # trained_feature_extractor = load_trained_feature_extractor_main(model_name=model_name, num_features=num_features,
    #                                                                 epoch_num=read_epoch_num, weights_dir=weights_dir)
    # main_cifar10(feature_extractor=trained_feature_extractor)

    print("Done :)")

