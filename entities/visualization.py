import numpy as np
import torch
import clip
import matplotlib.pyplot as plt


class VisualizeSimilarityMatrix:
    def __init__(self, clip_model, feature_extractor=None):
        self.model = clip_model
        self.extractor = feature_extractor

    def cifar10(self, images, texts, images_dict):
        # Building features
        image_input = torch.tensor(np.stack(images))
        text_tokens = clip.tokenize(["This is a " + desc for desc in texts])

        with torch.no_grad():
            if self.extractor is None:
                image_features = self.model.encode_image(image_input).float()
            else:
                image_features = self.extractor(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Calculating cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        count = len(images_dict)

        # Plotting the similarity matrix
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)

        # Plotting the captions as yticks
        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])

        # Plotting the tiny images above
        for i, image in enumerate(images):
            image = np.transpose(image.numpy(), (1, 2, 0))  # convert from Tensor image and transpose
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        # Writing the similarity values in the middle
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
        plt.show()

    def skimage(self, images, texts, descriptions, original_images):

        image_input = torch.tensor(np.stack(images))
        text_tokens = clip.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            if self.extractor is None:
                image_features = self.model.encode_image(image_input).float()
            else:
                image_features = self.extractor(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Calculating cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        count = len(descriptions)

        # ploting the similarity matrix
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        # plt.colorbar()

        # ploting the captions as yticks
        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])

        # ploting the tiny images above
        for i, image in enumerate(original_images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        # Writing the similarity values in the middle
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
        plt.show()
