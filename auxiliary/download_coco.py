import os
import urllib.request
import zipfile


def download_coco_dataset(target_dir='data/coco'):
    # URLs for the zip files
    image_url = 'http://images.cocodataset.org/zips/train2014.zip'
    annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

    # Make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Paths to the zip files
    image_zip_path = os.path.join(target_dir, 'train2014.zip')
    annotation_zip_path = os.path.join(target_dir, 'annotations_trainval2014.zip')

    # Check if images are already downloaded
    if not os.path.exists(image_zip_path):
        print('Downloading images...')
        urllib.request.urlretrieve(image_url, image_zip_path)
        print('Image download complete!')

    # Check if annotations are already downloaded
    if not os.path.exists(annotation_zip_path):
        print('Downloading annotations...')
        urllib.request.urlretrieve(annotation_url, annotation_zip_path)
        print('Annotation download complete!')

    # Paths to the unzipped directories
    image_dir = os.path.join(target_dir, 'images')
    annotation_dir = os.path.join(target_dir, 'annotations')

    # Extract images if not already extracted
    if not os.path.exists(image_dir):
        print('Extracting images...')
        with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print('Image extraction complete!')

    # Extract annotations if not already extracted
    if not os.path.exists(annotation_dir):
        print('Extracting annotations...')
        with zipfile.ZipFile(annotation_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print('Annotation extraction complete!')


if __name__ == "__main__":
    download_coco_dataset()
