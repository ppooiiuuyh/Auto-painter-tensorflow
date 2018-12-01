import sys

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt

def hatch(image):
    """
    A naive hatching implementation that takes an image and returns the image in
    the style of a drawing created using hatching.
    image: an n x m single channel matrix.
  
    returns: an n x m single channel matrix representing a hatching style image.
    """
    xdogImage = xdog(image, 0.1)
    
    hatchTexture = cv2.imread('./textures/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    height = len(xdogImage)
    width = len(xdogImage[0])
    
    if height > 1080 or width > 1920:
        print ("This method only supports images up to 1920x1080 pixels in size")
        sys.exit(1)
    
    croppedTexture = hatchTexture[0:height, 0:width]
    
    return xdogImage + croppedTexture


def xdog(image, epsilon=0.01, sigmaList = [0.3]):
    """
    Computes the eXtended Difference of Gaussians (XDoG) for a given image. This
    is done by taking the regular Difference of Gaussians, thresholding it
    at some value, and applying the hypertangent function the the unthresholded
    values.
    image: an n x m single channel matrix.
    epsilon: the offset value when computing the hypertangent.
    returns: an n x m single channel matrix representing the XDoG.
    """
    phi = 10**9
    
    difference = dog(image) / 255
    diff = difference * image
    
    for i in range(0, len(difference)):
        for j in range(0, len(difference[0])):
            if difference[i][j] >= epsilon:
                difference[i][j] = 1
            else:
                ht = np.tanh(phi * (difference[i][j] - epsilon))
                difference[i][j] = 1 + ht
    
    return difference * 255


def dog(image, k=4.5, gamma=0.95, sigmaList = [0.3]):
    """
    Computes the Difference of Gaussians (DoG) for a given image. Returns an image
    that results from computing the DoG.
    image: an n x m array for which the DoG is computed.
    k: the multiplier the the second Gaussian sigma value.
    gamma: the multiplier for the second Gaussian result.
  
    return: an n x m array representing the DoG
    """
    
    s1 = np.random.choice(sigmaList,1)
    s2 = s1 * k
    
    gauss1 = gaussian_filter(image, s1)
    gauss2 = gamma * gaussian_filter(image, s2)
    
    differenceGauss = gauss1 - gauss2
    return differenceGauss


if __name__ == '__main__':
    image_path = "./../../datasets/autocolorization/illustrations_resized/130.png"
    image = misc.imread(image_path, mode='RGB')
    # Main method defaults to computiong the eXtended Diference of Gaussians
    image = np.mean(image,axis=-1)
    #image = ((255 - image) > (255-150))
    #image = 255- (((255- image)>(255-10)) * (255-image))

    # image = image[:, :, 2]

    image = xdog(image)
    #result = 255- (((255- result)>(255-50)) * (255-result))
    print(image)
    plt.imshow(image,cmap="gray")
    plt.show()