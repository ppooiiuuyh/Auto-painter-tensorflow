import os
from scipy import misc
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from tqdm import tqdm, tqdm_gui

def func1():
    #dataset_name = "edges2shoessmall"
    dataset_name = "mapssmall"
    input_list = sorted(glob('./dataset/{}/*.*'.format(dataset_name + '/data')))
    print(input_list)
    print(len(input_list))
    
    check_folder('./dataset/{}/'.format(dataset_name + '/trainA'))
    check_folder('./dataset/{}/'.format(dataset_name + '/trainB'))
    
    
    for e,data in tqdm(enumerate(input_list)):
        image = misc.imread(data, mode='RGB')
        misc.imsave('./dataset/{}/'.format(dataset_name + '/trainA') + str(e) +'.jpg', np.array(image)[:,0:image.shape[1]//2])
        misc.imsave('./dataset/{}/'.format(dataset_name + '/trainB') + str(e) +'.jpg', np.array(image)[:,image.shape[1]//2:])
    #trainB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))
    
    
def func2():
    input_list = sorted(glob('./../datasets/autocolorization/illustrations_resized/*.*'))

    check_folder('./../datasets/autocolorization/illustrations_resized_256/original/')
    check_folder('./../datasets/autocolorization/illustrations_resized_256/xdog/')

    for e, data in tqdm_gui(enumerate(input_list)):
        image = misc.imread(data, mode='RGB')
        image = resize_and_crop(image,256)
        misc.imsave('./../datasets/autocolorization/illustrations_resized_256/original/' + str(e) + '.png', np.array(image))
        image = xdog(image,sigma_list=[0.3,0.4,0.5])
        misc.imsave('./../datasets/autocolorization/illustrations_resized_256/xdog/' + str(e) + '.png', np.array(image))


if __name__ == "__main__":
    func2()