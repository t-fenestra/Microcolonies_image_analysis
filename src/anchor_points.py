import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import skimage.io as io
from skimage import util

from glob import glob
import ast

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def show_size_bar(ax,traj_seg):
    dx= 0.810 # 1px in mkm
    length=10 # px
    nframes,row,col=traj_seg.shape
    
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-30
    start_y=row-15
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x,start_y-3, label,color='cyan')
    
    
def object_anchor_area(traj,binary_images,ax):
    """ calculate overlapped area i.e sum of pixels that are stay the same across time """
    
    min_row,min_col=np.min(traj.loc[:,'bbox-0':'bbox-1'])
    max_row,max_col=np.max(traj.loc[:,'bbox-2':'bbox-3'])
    
    time_start=np.min(traj.frame)
    time_end=np.max(traj.frame)
    binary_img_region=binary_images[time_start:time_end,min_row:max_row,min_col:max_col]
    
    # calculate anchor mask
    anchor_points=np.zeros_like(binary_img_region[0],dtype=np.int32)
    anchor_points=np.sum(binary_img_region,axis=0)
    anchor_mask=anchor_points>=(255*(binary_img_region.shape[0]))
    
    # overlap
    sum_overlap_px=np.sum(anchor_mask)
    overlapped_area_ratio=sum_overlap_px/np.mean(traj.area)
    
    
    ax.imshow(anchor_points,vmin=254,vmax=np.max(anchor_points),cmap='plasma')
    ax.axis("off")
    if np.sum(anchor_mask)>0:
        ax.set_title("particle_id={} \n anchored=True \n overlapped_area={:.2f}".format(traj["particle"].iloc[0],overlapped_area_ratio))
    else:
        ax.set_title("particle_id={} \n anchored=False".format(traj["particle"].iloc[0]))
    #print(traj.particle.iloc[0],np.mean(traj.area),sum_overlap_px)
    
    
    show_size_bar(ax,binary_img_region)
    
    result={'particle':traj.particle.iloc[0],
            'mean_area':np.mean(traj.area),
            'overlapped_area':sum_overlap_px,
            'overlapped_area_ratio':overlapped_area_ratio,
            'traj_len':traj.shape[0]}
    
    return result 






def plot_anchor_points(binary_images,tracks,file_prefix):
    from matplotlib.backends.backend_pdf import PdfPages
    sns.set(font_scale=1.5)

    # create figures
    Ngroups=tracks.groupby('particle').ngroups
    Nrows=np.int(np.ceil(Ngroups/5))
    
    # How many pages in pdf?
    Npages=Nrows//8
    add_extra_page=Nrows%8 
    Npages=Npages+add_extra_page
    print(Ngroups)
    pdf_file_name="../results/anchor_area_projection_plot/Projection_plot_{}.pdf".format(file_prefix)
    
    with PdfPages(pdf_file_name) as pdf:
        counter=0
        Anchor_data=[]
        
        # initialize first page
        fig,ax=plt.subplots(nrows=8,ncols=5,figsize=(40,60))
        suptitle = plt.suptitle(file_prefix, y=1.02)
        plt.tight_layout()
        axx=ax.ravel()
        
        
        for id, traj in tracks.groupby('particle'):
            if (counter%40==0) and (counter!=0):
                # close previous page
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                counter=0
                
                # create new page
                fig,ax=plt.subplots(nrows=8,ncols=5,figsize=(40,60))
                suptitle = plt.suptitle(file_prefix, y=1.02)
                plt.tight_layout()
                axx=ax.ravel()
                
                result=object_anchor_area(traj,binary_images,axx[counter])
                counter=counter+1
                Anchor_data.append(result)
            
            else:
                result=object_anchor_area(traj,binary_images,axx[counter])
                counter=counter+1
                Anchor_data.append(result)
        
        # last page
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


    Anchor_df=pd.DataFrame(Anchor_data)
    Anchor_df["type"]="fixed"
    Anchor_df["type"][Anchor_df["overlapped_area_ratio"]<=0.50]="rotating"
    Anchor_df["type"][Anchor_df["overlapped_area_ratio"]==0]="moving"
    Anchor_df['file_prefix']=file_prefix
    Anchor_df.to_csv("../results/anchor_area_csv/Anchor_area_{}.csv".format(file_prefix))
    #plt.savefig("../results/anchor_area_projection_plot/Projection_plot_{}.png".format(file_prefix), bbox_extra_artists=(suptitle,), bbox_inches="tight")
    
    
    return Anchor_df

