#!/usr/bin/env python
# coding: utf-8

# In[1]:


MODEL_PATH = 'architectures/'
DATASET_PATH = 'dataset/'
RANDOM_SEED = 42 # Set to `None` for the generator uses the current system time.


# In[2]:


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


# In[3]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

import time
import os
import requests
import shutil
from tqdm import tqdm

np.random.seed(RANDOM_SEED)


# In[4]:


def download_file(url, save_path, file_name=None, extract=False, force_download=False):
    """Download a file and optionally extract if it's compressed."""
    print(f"URL: {url}")
    
    if not file_name:
        file_name = os.path.basename(url)

    # If not forcing download and the file already exists, skip downloading.
    if not force_download and os.path.exists(os.path.join(save_path, file_name)):
        print(f"File `{os.path.join(save_path, file_name)}` already exists.")
        return

    os.makedirs(save_path, exist_ok=True)
    
    # Adjust file name if extraction is expected (e.g., assume download is a zip file)
    file_name += '.zip' if extract else ''
    file_path = os.path.join(save_path, file_name)
    
    # Send GET request in streaming mode
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"| Failed: [{response.reason}]")
        return response

    # Get the total size from the response header (if available)
    total_size = int(response.headers.get('Content-Length', 0))
    chunk_size = 1024  # 1KB chunks

    # Open the output file and use tqdm to show the progress bar.
    with open(file_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=f"Downloading `{file_name}`"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Extract the archive if requested
    if extract:
        try:
            shutil.unpack_archive(file_path, extract_dir=save_path)
            print(f"Extracted archive... ", end='')
            # Remove the archive file after extraction
            os.remove(file_path)
        except shutil.ReadError:
            print(f"| Note: {file_path} is not an archive or could not be extracted ", end='')
    
    print(f"| Succeeded")
    return response
    
def download_dataset_from_kaggle(file_name=None, save_path=DATASET_PATH, extract=True, force_download=False,
                                 dataset_url="datasets/download/kartik2112/fraud-detection"):
    # Download the dataset from Kaggle
    BASE_URL = "https://www.kaggle.com/api/v1"

    
    # Factor char list
    download_file(
        f'{BASE_URL}/{dataset_url}/{file_name}',
        save_path,
        file_name,
        extract,
        force_download
    )


# In[5]:


def feature_engineering(data):
    data = data.copy()
    
    #----------Cleaning----------
    # Remove null entries
    data.dropna(inplace=True)
    # Remove prefix `fraud-` in `merchant` feature
    data['merchant'] = data['merchant'].str.replace('fraud_', '')
    
    ## 1) Derive `age`, `age_group` feature from `dob`
    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = (pd.Timestamp.today() - data['dob']).dt.days // 365

    # Define age bins (from 18 to 90 in 10-year increments)
    age_bins = range(data['age'].min(), data['age'].max() + 11, 10)
    
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
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.to_list()
    data[categorical_features] = data[categorical_features].astype('category')

    return data


# In[6]:


def pre_processing(data, encoding=True, isTestSet=False):
    data = data.copy()
    
    # # Balancing the data (without SMOTE)
    # non_fraud = data[data['is_fraud'] == 0]
    # fraud = data[data['is_fraud'] == 1]
    # non_fraud = non_fraud.sample(len(fraud))
    # data = pd.concat([non_fraud, fraud])

    #  ----------Split up labels----------
    x = data.drop(["is_fraud"], axis=1).to_numpy()
    y = data["is_fraud"]

    categorical_features = {col: data.columns.get_loc(col) for col in data.select_dtypes('category').columns}
    int_features = {col: data.columns.get_loc(col) for col in data.select_dtypes(int).columns}
    float_features = {col: data.columns.get_loc(col) for col in data.select_dtypes(float).columns}
    age_group_order = data['age_group'].cat.categories
    
    # ----------Perform data encoding----------
    transformations = {}
    if encoding:
          
        # print(f'One Hot Encoding is applied for `{list(categorical_features.keys())}`')
        # data = pd.get_dummies(data, prefix=list(categorical_features.keys()), columns=list(categorical_features.keys()), dtype='category')
          
        print(f'Ordinal-Encoding is applied for `{list(categorical_features.keys())}`')
        for col, idx in categorical_features.items():
            le = LabelEncoder()
            x[:, idx] = le.fit_transform(x[:, idx])
            transformations[col] = le  # Store for future reference
    
    # ----------Standardization----------
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    transformations['scaler'] = scaler

    # ----------SMOTE technique----------
    # If the training data imbalanced we’ll address this using Synthetic Minority Oversampling Technique (SMOTE).
    # It is an oversampling technique that creates artificial minority class samples.
    # In our case, it creates synthetic fraud instances and so corrects the imbalance in our dataset.
    if not isTestSet and y.value_counts()[0] != y.value_counts()[1]:
        x, y = SMOTE().fit_resample(x, y)
        x, y = shuffle(x, y) # Then explicitly shuffle the data
        print('SMOTE is applied')
        
        # Upadte new synthetic data
        x_unscaled = transformations['scaler'].inverse_transform(x)
        x_unscaled = x_unscaled.astype(object)


        for col, idx in categorical_features.items():
            x_unscaled[:, idx] = transformations[col].inverse_transform(x_unscaled[:, idx].astype(int))
        
        data = pd.DataFrame(np.concatenate([x_unscaled, np.expand_dims(y, axis=1)], axis=1), columns=data.columns)
        data[list(categorical_features.keys())] = data[list(categorical_features.keys())].astype('category')
        data[list(int_features.keys())] = data[list(int_features.keys())].astype(int)
        data[list(float_features.keys())] = data[list(float_features.keys())].astype(float)
        data['age_group'] = data['age_group'].cat.reorder_categories(age_group_order, ordered=True)
          
    return x, y, data, transformations


# In[7]:


def load_models(models=[], model_path=MODEL_PATH):
    loaded_models = {}
    model_names = models
    
    print("\n===== MODEL METADATA =====")

    # Convert single values into lists for consistent processing
    if not isinstance(models, list):
        model_names = [models]
        
    # Get all available models if `model_names` is empty
    if not models or not models[0]:
        model_names = os.listdir(model_path)
        print(f"[INFO] Found [{len(model_names)}] models in {model_path}")
    
    for model_name in model_names:
        full_path = os.path.join(model_path, model_name)
        
        try:
            model = tf.keras.models.load_model(full_path)
            model._name = model_name

            # Print metadata
            print(f"\n=== Model: `{model_name}` ===")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Number of layers: {len(model.layers)}")
            print(f"Total parameters: {model.count_params():,}")
            print(f"File size: {os.path.getsize(full_path) / (1024 * 1024):.2f} MB")
            print(f"Last modified: {time.ctime(os.path.getmtime(full_path))}")
            
            print("\n" + "-"*50)

            loaded_models[model_name] = model
        except Exception as e:
            print(f"\n**Error loading model `{model_name}`: `{e}`")
            print("-"*50)

    if models and isinstance(models, str):
        loaded_models = loaded_models[models]
            
    return loaded_models


# In[8]:


def save_predictions(model_name, y_predict, predictions_csv=os.path.join(DATASET_PATH, 'predictions.csv')):
    """
    Save model predictions to a CSV file
    
    Args:
        model_name: Name of the model
        y_predict: Prediction array from model
        predictions_csv: Path to save/update predictions
    """
    print(f"\n[INFO] Saving model `{model_name}` predictions into `{predictions_csv}`...")
    
    try:
        predictions = pd.read_csv(predictions_csv)
        print(f"[INFO] Loaded `{os.path.basename(predictions_csv)}` file.")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        predictions = pd.DataFrame()
        print(f"[INFO] No `{os.path.basename(predictions_csv)}` found or file is empty. Creating a new file.")
    
    # Ensure y_predict is a 1D array and append to DataFrame
    predictions[model_name] = y_predict.flatten()
    predictions.to_csv(predictions_csv, index=False)
    print(f"[INFO] Saved model `{model_name}` predictions.")


# In[9]:


# Export this notebook into script `.py` file
# jupyter nbconvert --to script utils.ipynb

