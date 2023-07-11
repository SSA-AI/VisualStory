import clip
import torch


class CLIPProcessor:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def process_images(self, images):
        return self.model.encode_image(images)

    def generate_text(self, image_features):
        text = clip.tokenize(["A photo of a {label}."])
        text = text.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features
