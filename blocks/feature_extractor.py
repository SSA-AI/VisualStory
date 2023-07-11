import torch
from torchvision.models import resnet18

class FeatureExtractor:
    def __init__(self, device="cuda"):
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Removing the last layer (FC layer)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def extract_features(self, images):
        """
        Args:
            images (torch.Tensor): Tensor of images of shape (B, C, H, W)
        Returns:
            features (torch.Tensor): The extracted features of shape (B, F)
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model(images)  # Shape: [B, 512, H', W']
        features = features.view(features.size(0), -1)  # Shape: [B, F]
        return features.cpu()
