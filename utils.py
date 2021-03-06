# -*- coding: utf-8 -*-

import cv2, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from random import random
import matplotlib.pyplot as plt

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
INPUT_SHAPE_CROP = (IMAGE_HEIGHT-60-25, IMAGE_WIDTH, IMAGE_CHANNELS)
LOWER_IND = 60
UPPER_IND = 135

def remove_small_steering(data, thresh=0.05, drop_ratio=0.8):
    index = data[abs(data['steering'])<thresh].index.tolist()
    rows = [i for i in index if random() * drop_ratio]
    data = data.drop(data.index[rows])
    return data

def crop_img(img):
    return img[LOWER_IND:UPPER_IND,:]

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def image_augument_test(data_dir, image_file):
    

    fig,axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.tight_layout()

    image = load_image(data_dir, image_file)
    angle = 0

    axes[0,0].imshow(image)
    axes[0,0].set_title('Original Left, Angle=' + str(angle), fontsize=10)
    
    image, angle = random_flip(image, angle)
    axes[0,1].imshow(image)
    axes[0,1].set_title('Flip, Angle=' + str(angle), fontsize=10)
    
    image, angle = random_translate(image, angle, 100, 10)
    axes[0,2].imshow(image)
    axes[0,2].set_title('Transform, Angle=' + str(angle), fontsize=10)
    
    image = random_brightness(image)
    axes[1,0].imshow(image)
    axes[1,0].set_title('Brightness, Angle=' + str(angle), fontsize=10)
    
    image = crop(image)
    axes[1,1].imshow(image)
    axes[1,1].set_title('Crop, Angle=' + str(angle), fontsize=10)

    image = rgb2yuv(image)
    axes[1,2].imshow(image)
    axes[1,2].set_title('RGB2YUV, Angle=' + str(angle), fontsize=10)

    plt.show()
    
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT-60-25, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
