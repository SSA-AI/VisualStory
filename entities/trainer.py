import os
import itertools

import clip
import torch
from huggingface_hub.utils import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


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
