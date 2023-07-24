import torch

from entities.data_handlers import DataHandlerSKImage, DataHandlerCifar10, PreProcessSet
from entities.model_loaders import ClipLoader
from entities.trainer import LossObj, Trainer
from entities.visualization import VisualizeSimilarityMatrix


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
    #################################################
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
