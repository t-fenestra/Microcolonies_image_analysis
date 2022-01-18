import numpy as np
import pandas as pd
import os


import skimage.io as io
from skimage import util
from skimage.filters import gaussian

import warnings
warnings.filterwarnings('ignore')
from glob import glob




def img_characteristrics(img,seq_number,threshold,file_prefix):
    Dict=[]
    for frame in range(img.shape[0]):
        dict_result={'file_name':file_prefix,
                     'seq_number':seq_number,
                     'frame':frame,
                     'im_5%quantile':np.quantile(img[frame],0.05),
                    'im_95%quantile':np.quantile(img[frame],0.95),
                    'im_med':np.median(img[frame]),
                    'im_max':np.max(img[frame]),
                    'im_min':np.min(img[frame]),
                    'threshold':threshold}
        Dict.append(dict_result)
    DF=pd.DataFrame(Dict)
    return DF