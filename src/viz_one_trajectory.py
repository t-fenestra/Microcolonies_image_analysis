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



def plot_density_plot(traj_seg,ax):
    anchor_map=np.sum(traj_seg,axis=0)
    ax.imshow(anchor_map,vmin=255,vmax=np.max(anchor_map),cmap='plasma')

def show_size_bar(ax,traj_seg):
    dx= 0.810 # 1px in mkm
    length=10 # px
    nframes,row,col=traj_seg.shape
    
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-30
    start_y=row-15
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x,start_y-3, label,color='cyan')
    
    
    
def one_trajectory_viz(images,images_binary,tracks,file_prefix,particle):
    
    ###========================================####
    traj=tracks[tracks["particle"]==particle]
    
    # make box bigger
    delta=15
    min_row,min_col=np.min(traj.loc[:,'bbox-0':'bbox-1'])
    max_row,max_col=np.max(traj.loc[:,'bbox-2':'bbox-3'])

    start_frame=np.min(traj.loc[:,'frame'])
    end_frame=np.max(traj.loc[:,'frame'])

    min_row=np.max([min_row-delta,0])
    min_col=np.max([min_col-delta,0])
    max_row=np.min([max_row+delta,images[0].shape[0]])
    max_col=np.min([max_col+delta,images[0].shape[1]])
               
    traj_img=images[start_frame:end_frame,min_row:max_row,min_col:max_col]
    traj_seg=images_binary[start_frame:end_frame,min_row:max_row,min_col:max_col]
    

    import matplotlib.animation as animation
    sns.set(font_scale=1.5)
    fps = 10
    nFrames=traj_seg.shape[0]


    ###===== init plot ==============================#
    # First set up the figure, the axis, and the plot element we want to animate
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5),constrained_layout=True)
    plt.suptitle("experiment={} \n particle_id={}".format(file_prefix,particle))
    # particle image
    im1=ax[0].imshow(traj_img[0],cmap='gray')
    im2=ax[1].imshow(traj_seg[0],cmap='gray')
    plot_density_plot(traj_seg,ax[2])

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    ax[0].set_title("original")
    ax[1].set_title("segmented")
    ax[2].set_title("density plot")

    show_size_bar(ax[0],traj_seg)
    show_size_bar(ax[1],traj_seg)
    show_size_bar(ax[2],traj_seg)
        

    ####======================================#
    def animate_func(i):
        im1.set_data(traj_img[i])
        im2.set_data(traj_seg[i])
        return [im1,im2]

    anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nFrames-1,
                               interval = 1000 / fps/100, # in ms
                               )

    anim.save('../results/video_one_trajectory/{}/{}_TrajectoryID_{}.mp4'.format(file_prefix,file_prefix,particle), fps=fps, extra_args=['-vcodec', 'libx264'])

# print('Done!')