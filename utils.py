import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc, ndimage
from scipy.ndimage.filters import gaussian_filter
import os, random
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======================================================================================================================
# Dataset
# ======================================================================================================================
def preprocessing(x):
    x = (x/255 - 0.5)*2 # -1 ~ 1
    return x


def prepare_data(path, size, len=-1,color=True):
    #input_list = sorted(glob('./dataset/{}/*.*'.format(dataset_name + '/trainA')))[0:len]
    #target_list = sorted(glob('./dataset/{}/*.*'.format(dataset_name + '/trainB')))[0:len]
    input_list = sorted(glob(path+'/*.*'))[0:len]

    images = []

    if color  :
        for image in tqdm(input_list):
            images.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))

        # trainA = np.repeat(trainA, repeats=3, axis=-1)
        # trainA = np.array(trainA).astype(np.float32)[:, :, :, None]

    else :
        for image in tqdm(input_list):
            images.append(np.expand_dims(misc.imresize(misc.imread(image, mode='L'), [size, size]), axis=-1))


    images = preprocessing(np.asarray(images))

    return images



def resize_and_crop(img, size):
    """ resize img """
    if img.shape[0] > img.shape[1]:
        sizes = [int(img.shape[0] / img.shape[1] * size), size]
    else:
        sizes = [size, int(img.shape[1] / img.shape[0] * size)]
    resized_img = misc.imresize(img, sizes)
    
    """ random crop """
    start_0 = random.randrange(0, (resized_img.shape[0] - size) + 1)
    start_1 = random.randrange(0, (resized_img.shape[1] - size) + 1)
    print(start_0, start_1)
    cropped_img = resized_img[start_0:start_0 + size, start_1:start_1 + size]
    
    return cropped_img




def random_touch(imgA,imgB, num_points, dia):
    """ analyze img """
    assert imgA.dtype == imgB.dtype and all(np.equal(imgA.shape,imgB.shape))
    
    if imgA.dtype == np.uint8:
        img_max = 255
    else:
        img_max = 1.0
    
    """ set white template """
    img_touched = np.copy(imgA) #np.zeros_like(imgB) + img_max
    
    """ blur image """
    img_filtered = ndimage.gaussian_filter(imgB, sigma=0.3)
    
    """ pick random points """
    points = np.array([[random.randrange(0, imgB.shape[0]) for y in range(num_points)],
                       [random.randrange(0, imgB.shape[1]) for x in range(num_points)]]).T
    
    """ touch points """
    for p in points:
        for i in range(p[0] - dia, p[0] + dia):
            if i > 0 and i < img_filtered.shape[0]:
                for j in range(p[1] - dia, p[1] + dia):
                    if j > 0 and j < img_filtered.shape[1]:
                        #if (i - p[0]) ** 2 + (j - p[1]) ** 2 < dia ** 2:
                        img_touched[i, j] = img_filtered[p[0], p[1]]
    
    return img_touched





def load_test_data(image_path, size=256, gray_to_RGB=False):
    if gray_to_RGB :
        img = misc.imread(image_path, mode='L')
        img = misc.imresize(img, [size, size])
        img = np.expand_dims(img, axis=-1)
    else :
        img = misc.imread(image_path, mode='RGB')
        img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img



def shuffle(x, y, seed = np.random.random_integers(low=0, high=1000)) :
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y




def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image


def xdog(image, epsilon=0.01, k=4.5, gamma=0.95, sigma_list=[0.3]):
    """
    Computes the eXtended Difference of Gaussians (XDoG) for a given image. This
    is done by taking the regular Difference of Gaussians, thresholding it
    at some value, and applying the hypertangent function the the unthresholded
    values.
    image: an n x m single channel matrix.
    epsilon: the offset value when computing the hypertangent.
    returns: an n x m single channel matrix representing the XDoG.
    """
    if len(image.shape) >2:
        image = np.mean(image,axis=-1)
        
    phi = 10 ** 9
    
    difference = dog(image,k,gamma, sigma_list= sigma_list) / 255
    diff = difference * image
    
    for i in range(0, len(difference)):
        for j in range(0, len(difference[0])):
            if difference[i][j] >= epsilon:
                difference[i][j] = 1
            else:
                ht = np.tanh(phi * (difference[i][j] - epsilon))
                difference[i][j] = 1 + ht
    
    return difference * 255


def dog(image, k=4.5, gamma=0.95, sigma_list = [0.3]):
    """
    Computes the Difference of Gaussians (DoG) for a given image. Returns an image
    that results from computing the DoG.
    image: an n x m array for which the DoG is computed.
    k: the multiplier the the second Gaussian sigma value.
    gamma: the multiplier for the second Gaussian result.

    return: an n x m array representing the DoG
    """
    
    s1 = np.random.choice(sigma_list,1)[0]
    s2 = s1 * k
    
    gauss1 = gaussian_filter(image, s1)
    gauss2 = gamma * gaussian_filter(image, s2)
    
    differenceGauss = gauss1 - gauss2
    return differenceGauss


# ======================================================================================================================
# Image
# ======================================================================================================================
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img





# ======================================================================================================================
# Others
# ======================================================================================================================
def show_all_variables():
    model_vars = tf.global_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def show_variables(scope,print_infor = True):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    slim.model_analyzer.analyze_vars(model_vars, print_info=print_infor)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir