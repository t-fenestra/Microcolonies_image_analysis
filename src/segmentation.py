#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smmoth image by small gaussian (1.5px)kernel
Substract background. Background estimated by large gauss kernel (150 px)
We saved intermediate images:

denoised -small gauss kernel
background - large gauss kernel
prepared -final denoised image

@author: pichugina
"""

import numpy as np
import pandas as pd
import os

import skimage.io as io
from skimage import util

from glob import glob
from skimage import measure
from skimage.segmentation import clear_border

def outsu_threshold(images_prepared):
    from skimage.filters import threshold_otsu
    """
    create list of images that used for multiprocessing
    """ 
    images_binary=np.zeros_like(images_prepared,dtype=np.ubyte)
    threshold=threshold_otsu(images_prepared)
    images_binary[images_prepared>threshold]=255
    
    ### clear objects toutching the boader
    for frame in range(images_binary.shape[0]):
        images_binary[frame]=clear_border(images_binary[frame])
    
   
    
    return images_binary,threshold