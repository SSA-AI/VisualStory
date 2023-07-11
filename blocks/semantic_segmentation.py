import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import functional as F


class SemanticSegmentation:
    def __init__(self, device="cuda"):
        # self.model = deeplabv3_resnet101(pretrained=True)
        self.model = fcn_resnet50(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def segment(self, images):
        """
        Args:
            images (torch.Tensor): Tensor of images of shape (B, C, H, W)
        Returns:
            output (torch.Tensor): The segmented images of shape (B, H, W)
        """
        with torch.no_grad():
            images = images.to(self.device)
            output = self.model(images)['out']  # Shape: [B, 21, H, W]
        output_predictions = output.argmax(1)  # Shape: [B, H, W]
        return output_predictions.cpu()
