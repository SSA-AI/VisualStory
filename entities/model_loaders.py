import os

import clip
import numpy as np
import torch
import torchvision
from torch import nn


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
        self._trained_fc_layer = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LoadTrainedModel:
    def __init__(self, model, model_dir):
        self.model = model
        self.model_dir = model_dir
        self.model._trained_fc_layer = True

    def load_weights(self, epoch_num=None):
        if epoch_num is None:
            epoch_num = self.find_latest_train()
            print("Loading latest train")
        model_path = os.path.join(self.model_dir, f'feature_extractor_epoch_{epoch_num}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file exists for epoch {epoch_num}")
        self.model.load_state_dict(torch.load(model_path))

    def find_latest_train(self):
        # Get list of all saved model files
        files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        # Extract epoch numbers from file names
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in files]
        # Return the maximum epoch number, i.e., the latest one
        return max(epochs)
