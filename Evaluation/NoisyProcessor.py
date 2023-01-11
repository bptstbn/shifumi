import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def remove_parts_transformation(image_path):
    # Read in the image
    image = cv2.imread(image_path)

    # Generate a random mask
    mask = np.random.rand(image.shape[0], image.shape[1]) < 0.25 # 0.25 is the probability of a pixel to be removed
    image[mask] = [0, 0, 0]
    return image


def apply_gaussian_noise(image_path, std=25, black_and_white=True):
    # Read in the image
    image = cv2.imread(image_path)

    if black_and_white:
        # Convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Generate random noise
    noise = np.zeros(image.shape)
    cv2.randn(noise, 0, std)
    image = image + noise

    # clip the pixel values of the image
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')

    if black_and_white:
        # Convert back to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def show_image_before_after_rm(image_path, prob):
    """
    Shows image before and after random removal of pixels.

    @param image_path: path of image
    @param prob: probability of removing a pixel; between 0 and 1; higher means more noise
    @return: shows two images (before and after)
    """

    # Read in the image
    image = cv2.imread(image_path)

    # Create a copy of the image to be modified
    image_cpy = image.copy()

    # Generate a random mask
    mask = np.random.rand(image_cpy.shape[0], image_cpy.shape[1]) < prob
    image_cpy[mask] = [0, 0, 0]

    # Plot the original and modified images
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image[:, :, ::-1])
    ax1.set_title("Original Image")
    ax2.imshow(image_cpy[:, :, ::-1])
    ax2.set_title("Modified Image")
    plt.show()


def show_image_before_after_noise(image_path, std=255, black_and_white=True):
    """
    Shows image before and after gaussian noise.

    @param image_path: path of image
    @param std: higher standard deviation = more noise; greyscale pictures need a lower value
    @param black_and_white: makes sure output is greyscale
    @return: shows two images (before and after)
    """

    # Read in the image
    image = cv2.imread(image_path)

    # Create a copy of the image to be modified
    image_cpy = image.copy()

    if black_and_white:
        # Convert image to grayscale
        image_cpy = cv2.cvtColor(image_cpy, cv2.COLOR_RGB2GRAY)

    # Generate random noise
    noise = np.zeros(image_cpy.shape)
    cv2.randn(noise, 0, std)
    image_cpy = image_cpy + noise

    # clip the pixel values of the image
    image_cpy = np.clip(image_cpy, 0, 255)
    image_cpy = image_cpy.astype('uint8')

    if black_and_white:
        # Convert back to RGB
        image_cpy = cv2.cvtColor(image_cpy, cv2.COLOR_GRAY2RGB)

    # Plot the original and modified images
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image[:, :, ::-1])
    ax1.set_title("Original Image")
    ax2.imshow(image_cpy[:, :, ::-1])
    ax2.set_title("Modified Image")
    plt.show()


def process_images_in_folder(input_folder: str, output_folder: str, image_transform):
    """
    uses image transformation on all pictures in a folder (and its sub-folders)
    :param input_folder: root folder of input images
    :param output_folder: root folder (and name) of output folder where all processed images will be stored
    :param image_transform: image transformation of choice
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):
                input_path = Path(subdir) / file
                output_path = output_folder / Path(subdir).relative_to(input_folder) / file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                input_img = image_transform(str(input_path))
                cv2.imwrite(os.path.join(output_path), input_img)


if __name__ == '__main__':

    # single picture

    #data_path = "datasets/testset/paper/paper5.png"
    #show_image_before_after_rm(data_path, 0.25)
    #show_image_before_after_noise(data_path, 25, True)
    #show_image_before_after_noise(data_path, 255, False)

    # folder

    process_images_in_folder("datasets/testset", "datasets/testset_rm_noise", remove_parts_transformation)
    process_images_in_folder("datasets/testset", "datasets/testset_gaussian_noise", apply_gaussian_noise)

