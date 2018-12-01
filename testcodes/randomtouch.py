import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import random



def random_touch(img, num_points, dia):
    """ analyze img """
    if img.dtype == np.uint8 : img_max = 255
    else    : img_max = 1.0
    
    """ set white template """
    img_touched = np.zeros_like(img) + img_max

    """ blur image """
    img_filtered = ndimage.gaussian_filter(img,sigma=0.3)

    """ pick random points """
    points = np.array([[ random.randrange(0,img.shape[0]) for y in range(num_points) ], [  random.randrange(0,img.shape[1]) for x in range(num_points)]]).T

    """ touch points """
    for p in points:
        for i in range(p[0]-dia, p[0]+dia):
            if i > 0 and i  < img_filtered.shape[0]:
                for j in range(p[1]-dia, p[1]+dia):
                    if j  > 0 and j  < img_filtered.shape[1]:
                        if (i-p[0])**2 + (j-p[1])**2 < dia**2:
                            img_touched[i,j] = img_filtered[p[0],p[1]]

    return img_touched


image_path = "./dataset/edges2shoessmall/trainB/3.png"
image = misc.imread(image_path, mode='RGB')

image = random_touch(image,5,5)

plt.imshow(image)
plt.show()