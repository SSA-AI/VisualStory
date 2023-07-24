import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class COCODataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, num_images=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            captions_file (string): File with captions annotations.
            transform (callable, optional): Optional transform to be applied on an image.
            num_images (int, optional): Number of images to load for debugging.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = num_images

        with open(captions_file, 'r') as f:
            self.captions = json.load(f)['annotations']

        if self.num_images is not None:
            self.captions = self.captions[:self.num_images]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_id = self.captions[idx]['image_id']
        img_name = os.path.join(self.root_dir, f'COCO_train2014_{str(img_id).zfill(12)}.jpg')
        image = Image.open(img_name).convert('RGB')
        caption = self.captions[idx]['caption']

        if self.transform:
            image = self.transform(image)

        return image, caption


