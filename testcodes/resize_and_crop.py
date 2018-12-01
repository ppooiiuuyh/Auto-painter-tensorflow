import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import random


def resize_and_crop(img, size):
    """ resize img """
    if img.shape[0] > img.shape[1] : sizes = [int(img.shape[0]/img.shape[1]*size),size]
    else : sizes = [size,int(img.shape[1]/img.shape[0]*size)]
    resized_img = misc.imresize(img, sizes)

    """ random crop """
    start_0 = random.randrange(0,( resized_img.shape[0] - size ) + 1)
    start_1 = random.randrange(0,( resized_img.shape[1] - size ) + 1)
    print(start_0,start_1)
    cropped_img = resized_img[start_0:start_0+size, start_1:start_1+size]
    
    return cropped_img


image_path = "./../../datasets/autocolorization/illustrations_resized/1.png"
image = misc.imread(image_path, mode='RGB')

image = resize_and_crop(image, 512)

plt.imshow(image)
plt.show()


