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


def plot_anchor_points(binary_images,tracks,file_prefix):
    # create folder where density regions will be save
    # late it will be used for one trajectory video assemble
    folder_to_save_region="../results/anchor_area_density_plot/"+file_prefix
    os.mkdir(folder_to_save_region)
    
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_file_name="../results/anchor_area_density_plot_pdf/Projection_plot_{}.pdf".format(file_prefix)
    sns.set(font_scale=1.5)

    # create figures
    Ngroups=tracks.groupby('particle').ngroups
    print("number tracked objects",Ngroups)
    Nrows=np.int(np.ceil(Ngroups/5))
    
    # How many pages in pdf?
    Npages=Nrows//8
    add_extra_page=Nrows%8 
    Npages=Npages+add_extra_page
        
    
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
                
                result=object_anchor_area(traj,binary_images,axx[counter],folder_to_save_region)
                counter=counter+1
                Anchor_data.append(result)
            
            else:
                result=object_anchor_area(traj,binary_images,axx[counter],folder_to_save_region)
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
    return Anchor_df




def object_anchor_area(traj,binary_images,ax,folder_to_save):
    """ calculate overlapped area i.e sum of pixels that are stay the same across time """
    
    ### create mask to take into account only pixels belong to the object of interest ###
    binary_img_region_mask=np.zeros_like(binary_images)
    for id in traj.index:
        cmin_row,cmin_col,cmax_row,cmax_col=traj.loc[id,'bbox-0':'bbox-3']
        frame=traj.loc[id,'frame']
        binary_img_region_mask[frame,np.int(cmin_row):np.int(cmax_row),np.int(cmin_col):np.int(cmax_col)]=1
    
    ### calculating anchor points anchor point is a pixel belong to object of interest that lighted across frames object showed in the image
    binary_img_applied_mask=binary_images*binary_img_region_mask
    time_start=np.min(traj.frame)
    time_end=np.max(traj.frame)
    density_plot=np.sum(binary_img_applied_mask,axis=0)
    anchor_point=density_plot>=(255*(time_end-time_start)) # frames number
    
    ### overlap metrics
    sum_overlap_px=np.sum(anchor_point)
    overlapped_area_ratio=sum_overlap_px/np.mean(traj.area)

    
    plot_anchor_density_plot(traj,density_plot,ax,sum_overlap_px,overlapped_area_ratio,folder_to_save)
    

    result={'particle':traj.particle.iloc[0],
            'mean_area':np.mean(traj.area),
            'overlapped_area':sum_overlap_px,
            'overlapped_area_ratio':overlapped_area_ratio,
            'traj_len':traj.shape[0]}
    
    
    return result 


def plot_anchor_density_plot(traj,density_plot,ax,sum_overlap_px,overlapped_area_ratio,folder_to_save):
    #select region boundaries 
    min_row,min_col=np.min(traj.loc[:,'bbox-0':'bbox-1'])
    max_row,max_col=np.max(traj.loc[:,'bbox-2':'bbox-3'])
    region=density_plot[min_row:max_row,min_col:max_col]
    ax.imshow(region,vmin=255,vmax=np.max(density_plot),cmap='plasma')
    show_size_bar(ax,region.shape[0],region.shape[1])
    
    ax.axis("off")
    if sum_overlap_px>0:
        ax.set_title("particle_id={} \n anchored=True \n overlapped_area {:.2f}".format(traj["particle"].iloc[0],overlapped_area_ratio))
    else:
        ax.set_title("particle_id={} \n anchored=False".format(traj["particle"].iloc[0]))
        
    io.imsave(folder_to_save+"/Density_"+str(traj["particle"].iloc[0])+'_'+folder_to_save.split('/')[-1]+".tiff",region)
        
def show_size_bar(ax,row,col):
    dx= 0.810 # 1px in mkm
    length=10 # px
 
    label="{0:.2f} micron".format(length*dx) # mm
    start_x=col-30
    start_y=row-15
    ax.plot([start_x,start_x+length],[start_y,start_y],color='cyan')
    ax.text(start_x,start_y-3, label,color='cyan')   

