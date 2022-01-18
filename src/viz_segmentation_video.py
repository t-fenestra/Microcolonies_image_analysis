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


import matplotlib.animation as animation
import matplotlib.patches as patches


def show_size_bar(ax,images_binary):
    frames,row,col=images_binary.shape
    dx= 0.810 # 1px in mkm
    length=500 # px
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-700
    start_y=row-200
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x+length/2-200,start_y-50, label,color='cyan')


def segmentation_check_video(images,images_denoised,images_binary,file_prefix):
    sns.set(font_scale=2.0)
    fps = 2
    nFrames=images.shape[0]

   ###===== init plot ==============================#
    # First set up the figure, the axis, and the plot element we want to animate
    frame=0
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(40,10),constrained_layout=True)
    plt.suptitle(file_prefix)

    # original
    im1=ax[0].imshow(images[frame],cmap="gray")
    ax[0].set_title("original")
    ax[0].axis("off")
    show_size_bar(ax[0],images)
    
    # denosed
    im2=ax[1].imshow(images_denoised[frame],cmap="gray")
    ax[1].set_title("img denoised")
    ax[1].axis("off")
    show_size_bar(ax[1],images)

    # binary
    im3=ax[2].imshow(images_binary[frame],cmap="gray")
    ax[2].set_title("img binary")
    ax[2].axis("off")
    show_size_bar(ax[2],images)
    


    ####===== animated fundtion ==============================#
    def animate_func(frame):
        im1.set_data(images[frame]) ### original ###
        im2.set_data(images_denoised[frame])   ### binary ###
        im3.set_data(images_binary[frame])   ### binary ###
        return [im1,im2,im3]

    anim = animation.FuncAnimation( fig, 
                                animate_func, 
                                frames = nFrames,
                                interval = 1000 / fps, # in ms
                                )



    anim.save('../results/video_segmentation/Experiment_{}.mp4'.format(file_prefix), fps=fps, extra_args=['-vcodec', 'libx264'])
