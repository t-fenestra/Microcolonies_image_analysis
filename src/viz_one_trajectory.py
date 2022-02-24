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



def plot_density_plot(file_prefix,ax,anchor_ratio,particle,dict_folders,type_object):
    anchor_map=io.imread(dict_folders["anchor_area_density_plot"]+"{}/Density_{}_{}.tiff".format(file_prefix,particle,file_prefix))
    ax.imshow(anchor_map,vmin=np.min(anchor_map),vmax=np.max(anchor_map),cmap='plasma')
    ax.set_title("type ={} \n anchor_ratio={:}".format(type_object,anchor_ratio))
    ax.axis("off")

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





def viz_trajectories(images_binary,tracks,frame,ax,min_row,min_col):
    """
    viz trajectories for particular frame
    """
    # show image
    ax.imshow(images_binary[frame],cmap='gray')
    ax.axis('off')
    ax.set_title("frame="+str(frame))
    show_size_bar(ax,images_binary)
        
    # viz tracks
    tracks_frame=tracks[tracks['frame']<=frame]
    for id,group in tracks_frame.groupby("particle"):
        ax.plot(group['x']-min_col,group['y']-min_row,lw=2,color='red')
        


def one_trajectory_viz_with_tracks(images_binary,traj,anchor_ratio,dict_folders,save_folder,file_prefix,type_object):
    
    ###========================================####
    ## cut box with trajectory
    ###=========================================####
    
    
    # make box bigger
    delta=15
    min_row,min_col=np.min(traj.loc[:,'bbox-0':'bbox-1'])
    max_row,max_col=np.max(traj.loc[:,'bbox-2':'bbox-3'])
    #print(min_row,min_col,max_row,max_col)

    start_frame=np.min(traj.loc[:,'frame'])
    end_frame=np.max(traj.loc[:,'frame'])

    min_row=np.max([min_row-delta,0])
    min_col=np.max([min_col-delta,0])
    max_row=np.min([max_row+delta,images_binary[0].shape[0]])
    max_col=np.min([max_col+delta,images_binary[0].shape[1]])
    
    #print(min_row,min_col,max_row,max_col)
    traj_seg=images_binary[:,min_row:max_row,min_col:max_col]


    ###===== init plot ==============================#
    # First set up the figure, the axis, and the plot element we want to animate
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5),constrained_layout=True)
    camera = Camera(fig)


    for frame in range(traj_seg.shape[0]):
        viz_trajectories(traj_seg,traj,frame,ax[0],min_row,min_col)
        plot_density_plot(file_prefix,ax[1],anchor_ratio,traj["particle"].iloc[0],dict_folders,type_object)
        camera.snap()
 
    #Creating the animation from captured frames
    animation = camera.animate(interval = 200, repeat = True,
                           repeat_delay = 500)

    #Saving the animation
    animation.save(save_folder+'/{}_{}_track_{}.mp4'.format(traj["particle"].iloc[0],type_object,file_prefix))
    
    
