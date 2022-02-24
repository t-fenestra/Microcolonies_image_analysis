import os
import glob

import numpy as np
import pandas as pd

import skimage.io as io
import skimage.util as util
from skimage.exposure import adjust_gamma
from skimage.segmentation import watershed
from skimage.filters import sobel,threshold_otsu

import scipy
from scipy import optimize
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
plt.style.use('default')

import seaborn as sns
import cv2
from celluloid import Camera


import matplotlib.animation as animation
import matplotlib.patches as patches


def show_size_bar(ax,img_dimensions):
    frames,row,col=img_dimensions
    dx= 0.810 # 1px in mkm
    length=500 # px
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-700
    start_y=row-200
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x+length/2-200,start_y-50, label,color='cyan')


def show_size_bar(ax,img_dimensions):
    frames,row,col=img_dimensions
    dx= 0.810 # 1px in mkm
    length=500 # px
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-700
    start_y=row-200
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x+length/2-200,start_y-50, label,color='cyan')


def segmentation_check_video(images,images_denoised,images_binary,dict_folders,file_prefix):
    sns.set(font_scale=2.0)
    img_dimensions=images_binary.shape
    ###===== init plot ==============================#
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(40,10),constrained_layout=True)
    plt.suptitle(file_prefix)
    camera = Camera(fig)
    

    for frame in range(images_binary.shape[0]):
        img_plot_ax(images[frame],ax[0],"original",img_dimensions)
        img_plot_ax(images_denoised[frame],ax[1],"denoised",img_dimensions)
        img_plot_ax(images_binary[frame],ax[2],"segmented",img_dimensions)
        camera.snap()
 
    #Creating the animation from captured frames
    animation = camera.animate(interval = 200, repeat = True,repeat_delay = 500)

    #Saving the animation
    animation.save(dict_folders["video_segmentation"]+'{}_segmentation.mp4'.format(file_prefix))
    
def img_plot_ax(img,ax,title,img_dimensions):
    ax.imshow(img,cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    show_size_bar(ax,img_dimensions)
