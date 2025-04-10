#!/usr/bin/env python
# coding: utf-8

# In[22]:


MODEL_PATH = 'architectures/'
DATASET_PATH = 'dataset/'
RANDOM_SEED = 42 # Set to `None` for the generator uses the current system time.


# In[14]:


import sys
import os

# Check Python version
print(f"Python Version: `{sys.version}`")  # Detailed version info
print(f"Base Python location: `{sys.base_prefix}`")
print(f"Current Environment location: `{os.path.basename(sys.prefix)}`", end='\n\n')

import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
from tensorflow.config import list_physical_devices

print(f"Tensorflow version: `{tf.__version__}`")
print(f"CUDNN version: `{tf_build_info.build_info['cudnn_version']}`")
print(f"CUDA version: `{tf_build_info.build_info['cuda_version']}`")
print(f"Num GPUs Available: {len(list_physical_devices('GPU'))}")


# In[18]:


import pandas as pd
import numpy as np


# In[20]:


def feature_engineering(data):
    ## 1) Derive `age`, `age_group` feature from `dob`
    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = (pd.Timestamp.today() - data['dob']).dt.days // 365

    # Define age bins (from 18 to 90 in 10-year increments)
    age_bins = range(data['age'].min(), data['age'].max() + 10, 10)
    
    data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=[f"{i}-{i+9}" for i in age_bins[:-1]], right=False)

    ## 2) Derive `dist` feature from `lat`,`long` and `merch_lat`, `merch_long`
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371

        return c * r

    # Calculate distances and create new column
    data['dist'] = haversine_distance(
        data['lat'],
        data['long'],
        data['merch_lat'],
        data['merch_long']
    )

    return data


# In[ ]:





# In[17]:


# Export this notebook into script `.py` file
# jupyter nbconvert --to script utils.ipynb

