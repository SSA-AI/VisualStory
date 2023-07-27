# VisualStoryTelling

# Project: Visual and Textual Embedding Similarity using CLIP and ResNet

This project explores the fascinating intersection of **Computer Vision** and **Natural Language Processing** by leveraging the power of OpenAI's **CLIP** model and **ResNet** architectures. The primary objective is to investigate the cosine similarity between textual embeddings and image embeddings.

## Overview

1. **CLIP (Contrastive Language–Image Pretraining)**: CLIP is a model developed by OpenAI that connects vision and language in a way that pushes the boundaries of what AI can understand and recognize. It is trained on a variety of internet text paired with images. This allows the model to learn directly from data in a format that contains a preponderance of the world's knowledge. More about CLIP can be found [here](https://openai.com/research/clip/).

2. **ResNet (Residual Networks)**: ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. This project uses ResNet18 and ResNet50 as feature extractors for images. More about ResNet can be found [here](https://arxiv.org/abs/1512.03385).

## Methodology

The project follows these key steps:

1. **Textual Embeddings**: Generate textual embeddings using the CLIP model. The embeddings are created from the class labels of the CIFAR10 dataset.

2. **Image Embeddings**: Generate image embeddings using a feature extractor. The feature extractor is a ResNet model (ResNet18/50) trained on the CIFAR10 dataset.

3. **Cosine Similarity**: Calculate the cosine similarity between the textual embeddings and the image embeddings. This similarity score serves as a measure of how well the ResNet model has learned to correlate the images with their respective class labels.

4. **Training**: Train the ResNet model on pairs of images and their corresponding text embeddings from the CIFAR10 dataset. The objective of the training is to improve the cosine similarity between the image embeddings and the text embeddings.

5. **Similarity Matrix**: Finally, calculate the similarity matrix. This matrix provides a comprehensive view of how each class in the CIFAR10 dataset correlates with every other class based on the trained ResNet model.

## Usage

This project is designed to be flexible and easily adaptable for various use cases. It can be used as a starting point for any project that requires a deep understanding of the relationship between images and text.

## Conclusion

This project provides a robust framework for exploring and understanding the relationship between visual and textual data. By leveraging powerful models like CLIP and ResNet, it opens up new possibilities for research and applications in the field of AI.
