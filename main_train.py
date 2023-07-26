import argparse
import os
import yaml
import torch
from entities.trainer import Trainer
from entities.loss_obj import LossObj
from torch.utils.tensorboard import SummaryWriter
from entities.data_handlers import DataHandlerCifar10
from entities.model_loaders import ResNetXFeatureExtractor, ClipLoader


def parse_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    with open(config_file, 'r') as file:
        config_content = file.read()

    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    log_output_dir = os.path.join(results_dir, "tensorboard_log")
    os.makedirs(log_output_dir, exist_ok=True)
    writer = SummaryWriter(log_output_dir)

    # Log the content of the config file
    writer.add_text('Config', config_content)
    writer.close()

    return config


def get_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    return parser.parse_args()


def train_main():

    args = get_args()
    config = parse_config(args.config)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name_str = config['model_name_str']
    print(f"Loading feature extractor: {model_name_str}...")
    num_features = config["num_features"]
    feature_extractor = ResNetXFeatureExtractor(model_name_str, num_features).to(device)

    # Initialize ClipLoader
    clip_model_str = config['clip_model_str']
    print(f"Loading CLIP model: {clip_model_str}...")
    clip_loader, preprocess = ClipLoader(clip_model_str).load()


    # Create DataHandler instance and download data
    data_dir = config['data_dir']
    data_handler = DataHandlerCifar10(data_dir=data_dir, batch_size=config['batch_size'], val_split=config['val_split'])
    data_handler.download_and_save()

    # Split the dataset into training and validation sets
    train_loader, val_loader, classes = data_handler.split_train_val()


    # Initialize loss object
    loss_obj = LossObj('mse')

    # Initialize trainer
    trainer = Trainer(feature_extractor, clip_loader, loss_obj, device,
                      results_dir=config["results_dir"],
                      gamma=config["schedualer_gamma"],
                      step=config["schedualer_step"])

    # Initialize optimizer
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=config['lr'])

    # Train
    trainer.train(train_loader, train_loader, classes, optimizer, num_epochs=config['num_epochs'])


if __name__ == "__main__":

    #Run trainng protocol
    train_main()

    print("Done :-)")
