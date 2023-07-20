import torch
from torch import nn
from torchvision.models import resnet50
from torch.nn import functional as F
from PIL import Image
import clip
import numpy as np

"""
This code calculates the cosine similarity between each image and its corresponding caption, 
once using the CLIP model and once using a custom model built on top of ResNet50. 
The resulting matrices show how similar each image is to its caption according to each model.

Please note that the labels variable is not used in this example, as it's assumed that you 
have already ordered your images and captions such that they correspond to each other. 
If this is not the case, you may need to add code to match the images and captions correctly.

As before, the code does not include the training of the new model built on top of ResNet50. 
You may want to adjust the architecture or training procedure depending on your use case.
"""

############ Train #################
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, optimizer, device, writer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer

    def train(self, images, texts, clip_model, epochs=10):
        self.model.train()

        # Freeze ResNet parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(len(images)):
                # Get the inputs
                inputs = preprocess(Image.fromarray(np.transpose((images[i] / 2 + 0.5).numpy(), (1, 2, 0)))).unsqueeze(
                    0).to(self.device)
                labels = clip_model.encode_text(clip.tokenize([texts[i]]).to(self.device))

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(images)
            print(f'Epoch: {epoch + 1}, Loss: {epoch_loss}')

            # Log loss to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', epoch_loss, epoch)

        print('Finished Training')


####################################
# 1. Load the pre-trained CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

# 2. Load the pre-trained ResNet50 model
resnet_model = resnet50(pretrained=True).to("cuda")
resnet_model.eval()


# 3. Define a new model
class MyModel(nn.Module):
    def __init__(self, base_model, output_dim):
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.fc.in_features, output_dim)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# You need to match the output dimension of CLIP model which is 512
my_model = MyModel(resnet_model, output_dim=512).to("cuda")

# Get the CIFAR10 data
data_loader = CIFAR10DataLoader()
images, labels = data_loader.get_data()
texts = ["This is a frog.", "This is a truck.", "This is a deer.", "This is a car.", "This is a bird.",
         "This is a horse.", "This is a ship.", "This is a cat.", "This is a dog.",
         "This is an airplane."]  # Replace these with your actual captions

# Initialize the similarity matrices
clip_similarities = np.zeros((len(images), len(texts)))
my_model_similarities = np.zeros((len(images), len(texts)))

# 4. Compute similarities
for i in range(len(images)):
    # Convert images and text to CLIP-preprocessed tensors
    image_tensor = preprocess(Image.fromarray(np.transpose((images[i] / 2 + 0.5).numpy(), (1, 2, 0)))).unsqueeze(0).to(
        "cuda")
    text_tokens = clip.tokenize([texts[i]]).to("cuda")

    with torch.no_grad():
        # Compute image and text features with the CLIP model
        clip_image_features = clip_model.encode_image(image_tensor)
        clip_text_features = clip_model.encode_text(text_tokens)

        # Compute the image features with the new model
        my_model_image_features = my_model(image_tensor)

    # Compute similarities
    clip_similarities[i, i] = F.cosine_similarity(clip_image_features, clip_text_features, dim=-1).item()
    my_model_similarities[i, i] = F.cosine_similarity(my_model_image_features, clip_text_features, dim=-1).item()

# 5. Compare the two similarity matrices
print("CLIP similarities:\n", clip_similarities)
print("My model similarities:\n", my_model_similarities)
