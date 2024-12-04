import struct
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from improved_precision_recall import IPR, get_custom_loader

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        magic, numimages, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read the image data and reshape it into a 2D array (number of images, 28x28)
        images = np.fromfile(f, dtype=np.uint8).reshape(numimages, rows, cols)
    return images


def loadgenerated_images(folder_path):
    generated_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(folder_path, filename)).convert('L')  # Grayscale
            generated_images.append(np.array(image))
    return generated_images

if __name__ == '__main__':
    # Load the test data
    test_images = load_mnist_images('data/MNIST/MNIST/raw/t10k-images-idx3-ubyte')

    output_folder = 'test_images'


    # Create folder if doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Enregistrer les images
    for i in range(test_images.shape[0]):
        img = Image.fromarray(test_images[i].astype(np.uint8))  # Convertir le tableau NumPy en image
        img.save(os.path.join(output_folder, f'image{i}.png'))  # Enregistrer l'image sous forme de PNG


    ipr = IPR()
    # Compute precision and recall between real and fake images
    with torch.no_grad():
        # Compute manifold for real images
        ipr.compute_manifold_ref('samples')

        # Compute precision and recall for fake images
        precision, recall = ipr.precision_and_recall('test_images')

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')