import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from typing import List


class VisualizeStory:
    def __init__(self, images, texts):
        """
        Args:
            images (list): List of images.
            texts (list): List of corresponding texts.
        """
        self.images = images
        self.texts = texts

    def display(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for img, txt in zip(self.images, self.texts):
            ax.imshow(img)
            ax.title.set_text(txt)
            ax.axis('off')

            display(fig)

            # Wait for user input and then clear the output
            input("Press enter to continue...")
            clear_output(wait=True)
