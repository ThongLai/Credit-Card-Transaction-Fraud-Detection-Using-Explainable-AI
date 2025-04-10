#!/usr/bin/env python
# coding: utf-8

# In[9]:


MODEL_PATH = 'architectures/'
DATASET_PATH = 'dataset/'
RANDOM_SEED = 42 # Set to `None` for the generator uses the current system time.


# In[10]:


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


# In[11]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

np.random.seed(RANDOM_SEED)


# In[12]:


def feature_engineering(data):
    data = data.copy()
    
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

    # Move label columns back at the end
    data.insert(len(data.columns)-1, "is_fraud", data.pop("is_fraud"))

     # Drop unccessary columns
    drop_cols = ["trans_date_trans_time", "cc_num","first","last","street","city", "state", "zip","job", 'dob',"trans_num"]
    data.drop(columns=drop_cols, axis=1, inplace=True)

    # Convert categorical coulumns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.to_list()
    data[categorical_cols] = data[categorical_cols].astype('category')

    return data


# In[17]:


def pre_processing(data, encoding=True):
    data = data.copy()
    
    # # Balancing the data
    # non_fraud = data[data['is_fraud'] == 0]
    # fraud = data[data['is_fraud'] == 1]
    # non_fraud = non_fraud.sample(len(fraud))
    # data = pd.concat([non_fraud, fraud])

    # Split up labels
    x = data.drop(["is_fraud"], axis=1).to_numpy()
    y = data["is_fraud"]
    
    # Perform data encoding
    transformations = {}
    if encoding:
        categorical = {col: data.columns.get_loc(col) for col in data.select_dtypes('category').columns}
          
        # print(f'One Hot Encoding is applied for `{list(categorical.keys())}`')
        # data = pd.get_dummies(data, prefix=list(categorical.keys()), columns=list(categorical.keys()), dtype='category')
          
        print(f'Ordinal-Encoding is applied for `{list(categorical.keys())}`')
        for col, idx in categorical.items():
            le = LabelEncoder()
            x[:, idx] = le.fit_transform(x[:, idx])
            transformations[col] = le  # Store for future reference
    
    # Standardization
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    transformations['scaler'] = scaler
    
    # If the training data imbalanced we’ll address this using Synthetic Minority Oversampling Technique (SMOTE).
    # It is an oversampling technique that creates artificial minority class samples.
    # In our case, it creates synthetic fraud instances and so corrects the imbalance in our dataset.
    if y.value_counts()[0] != y.value_counts()[1]:
        x, y = SMOTE().fit_resample(x, y)
        x, y = shuffle(x, y) # Then explicitly shuffle the data
        print('SMOTE is applied')
        
        # Upadte new synthetic data
        x_unscaled = transformations['scaler'].inverse_transform(x)
        x_unscaled = x_unscaled.astype(object)


        for col, idx in categorical.items():
            x_unscaled[:, idx] = transformations[col].inverse_transform(x_unscaled[:, idx].astype(int))
        
        data = pd.DataFrame(np.concatenate([x_unscaled, np.expand_dims(y, axis=1)], axis=1), columns=data.columns)
        data[list(categorical.keys())] = data[list(categorical.keys())].astype('category')
        
          
    return x, y, data, transformations


# # Testing 

# In[14]:


# test_data = pd.read_csv(os.path.join(DATASET_PATH, 'fraudTest.csv'), index_col=0)
# test_data = feature_engineering(test_data)
# x, y, data, transformations = pre_processing(test_data)


# In[15]:


# Export this notebook into script `.py` file
# jupyter nbconvert --to script utils.ipynb

