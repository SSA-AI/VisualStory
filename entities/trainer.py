import argparse
import os
import yaml
import clip
import torch
import itertools
from torch import nn
from huggingface_hub.utils import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from entities.data_handlers import DataHandlerCifar10, DataHandlerSKImage
from entities.model_loaders import ResNetXFeatureExtractor, ClipLoader


class Trainer:
    def __init__(self, feature_extractor, clip_model, loss_obj, device, results_dir, gamma, step):
        self.feature_extractor = feature_extractor
        self.clip_model = clip_model
        self.loss_obj = loss_obj
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.training_results_dir = os.path.join(results_dir, "training_results")
        os.makedirs(self.training_results_dir, exist_ok=True)
        self.log_output_dir = os.path.join(results_dir, "tensorboard_log")
        os.makedirs(self.log_output_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_output_dir)  # TensorBoard writer
        self.gamma = gamma
        self.step = step

    def train(self, dataloader, val_dataloader, classes, optimizer, num_epochs=10, subset_size=100):

        # training phase:
        #################
        self.feature_extractor.train()

        # Initialize the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=self.step, gamma=self.gamma)

        for epoch in tqdm(range(num_epochs)):
            # Create an iterator from the dataloader and use itertools.islice to only get a subset of it
            data_iter = iter(dataloader)
            for i, (images, labels) in enumerate(itertools.islice(data_iter, subset_size)):
                print(f"Iteration: {i}")
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

            # validation phase:

            # Validation phase every 5 epochs
            mod = 1
            if epoch % mod == 0:
                    self.feature_extractor.eval()  # Set the model to evaluation mode
                    with torch.no_grad():  # Disable gradient computation
                        val_loss_total = 0
                        val_steps = 0
                        for i, (images, labels) in enumerate(val_dataloader):
                            images = images.to(self.device)
                            labels = labels.to(self.device)

                            # Convert numerical labels to text labels
                            text_labels = [classes[label] for label in labels]

                            # Generate image features using feature extractor
                            image_features = self.feature_extractor(images)

                            # Generate label features using CLIP
                            label_features = self.clip_model.encode_text(clip.tokenize(text_labels))

                            # Compute loss
                            val_loss = self.loss_obj.get_loss()(image_features, label_features)
                            val_loss_total += val_loss.item()
                            val_steps += 1

                        # Compute average validation loss for this epoch
                        avg_val_loss = val_loss_total / val_steps

                        # Log the average validation loss to TensorBoard
                        self.writer.add_scalar('Validation Loss', val_loss.item(), epoch)

                    self.feature_extractor.train()  # Set the model back to training mode

            # Save the model's weights after each epoch
            torch.save(self.feature_extractor.state_dict(), f'{self.training_results_dir}/feature_extractor_epoch_{epoch}.pth')

            # Step the learning rate scheduler
            scheduler.step()

        # Close the TensorBoard writer
        self.writer.close()


class LossObj:
    def __init__(self, loss_name='mse'):
        if loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError("Loss function not supported")

    def get_loss(self):
        return self.loss_func


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
