import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import skimage.io as io
from skimage import util

from glob import glob
import trackpy




def tracking(df_measurements,fname):
    
    df_measurements=df_measurements.rename(columns={'centroid-1':"x",'centroid-0':"y"})
    tracks = trackpy.link_df(df_measurements,search_range=50)

    # filter out trajectories less than 5 frames
    clean_tracks = trackpy.filter_stubs(tracks,threshold=5)
    return clean_tracks