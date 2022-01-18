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

def show_size_bar(ax,images_binary):
    frames,row,col=images_binary.shape
    dx= 0.810 # 1px in mkm
    length=500 # px
    label="{0:.2f} micron".format(length*dx) 
    start_x=col-700
    start_y=row-200
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x+length/2-200,start_y-50, label,color='cyan')
    
    
    
def viz_trajectories(images_binary,tracks,frame,ax):
    """
    viz trajectories for particular frame
    """
    # show image
    ax.imshow(images_binary[frame],cmap='gray')
    ax.axis('off')
    show_size_bar(ax,images_binary)
        
    # viz tracks
    tracks_frame=tracks[tracks['frame']<=frame]
    for id,group in tracks_frame.groupby("particle"):
        ax.plot(group['x'],group['y'],lw=2,color='red')
            

def trajectories_to_video(images_binary,tracks,save_folder,file_prefix):
    sns.set(font_scale=1.5)
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(20,20),constrained_layout=True)
    ax.set_title(file_prefix +'/n'+"tracks")
    camera = Camera(fig)


    for frame in range(images_binary.shape[0]):
        viz_trajectories(images_binary,tracks,frame,ax)
        camera.snap()
 
    #Creating the animation from captured frames
    animation = camera.animate(interval = 200, repeat = True,
                           repeat_delay = 500)

    #Saving the animation
    animation.save(save_folder+'/{}_tracking.mp4'.format(file_prefix))