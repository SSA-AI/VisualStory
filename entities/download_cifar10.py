import os
from torchvision.datasets import CIFAR10


def download_cifar10_dataset(target_dir='data/cifar10'):
    # Path to the CIFAR10 train and test data
    train_data_path = os.path.join(target_dir, 'cifar-10-batches-py/data_batch_1')
    test_data_path = os.path.join(target_dir, 'cifar-10-batches-py/test_batch')

    # Check if dataset is already downloaded
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print('Downloading CIFAR-10 dataset...')
        CIFAR10(root=target_dir, train=True, download=True)
        CIFAR10(root=target_dir, train=False, download=True)
        print('Download complete!')
    else:
        print('CIFAR-10 dataset already exists!')

if __name__ == "__main__":
    download_cifar10_dataset()
