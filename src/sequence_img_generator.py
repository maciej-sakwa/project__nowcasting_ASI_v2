from typing import Tuple

import pandas as pd
from datetime import datetime, timedelta
from pvlib import location, solarposition
import tensorflow as tf
from keras.utils import Sequence
from keras.layers import RandomFlip, RandomBrightness, RandomRotation
import numpy as np
import cv2
import os

"""
Data processing module. Contains definition of the classes and functions used to stack data.
V.1.0.0 MS 09/10/23

Class DataGenerator_SCNN(Sequence) - stacks images into 3 img sequence for prediction
    Init methods:
    |- __init__() - Initialization, dataframe with ['Irr', 'Image', 'Target'] columns is necessary input
    |- __len__() - Denotes the number of batches per epoch
    |- __getitem__() - Generate one batch of data
    Methods:
    |- on_epoch_end - Updates indexes after each epoch
    Private Methods:
    |- __data_generation(dataframe) - Generates data containing batch_size samples, 
                dataframe with ['Irr', 'Image', 'Target'] columns is necessary input
Class  DataGeneratorGHI_SCNN(Sequence) - stacks images into 3 img sequence for prediction (same as above but it encodes upper pixels in images for SCNN)
    Init methods:
    |- __init__() - Initialization, dataframe with ['Irr', 'Image', 'Target'] columns is necessary input
    |- __len__() - Denotes the number of batches per epoch
    |- __getitem__() - Generate one batch of data
    Methods:
    |- on_epoch_end - Updates indexes after each epoch
    Private Methods:
    |- __data_generation(dataframe) - Generates data containing batch_size samples, 
                dataframe with ['Irr', 'Image', 'Target'] columns is necessary input, encodes upper pixels with GHI info
Functions:
|- generate_dataframe(GHI_PATH, FORECAST_HORIZON, SEQUENCE_HORIZON = None, CLEAR_SKY_MODEL='simplified_solis') - Example of the dataframe generation funcion
"""


class DataGenerator_SCNN(Sequence):

    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=128, dim=(128, 128, 3), channel_IMG = 1, shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest
        
        self.rotate = tf.keras

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)): # stack images in channels
                img = cv2.imread(img_row[i_IMG], 0)
                if img is None:
                    raise ValueError(f'{img_row[i_IMG]} does not exist')
                else:
                    X[n_index, :, :, i_IMG] = img / 255.0
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y


class DataGeneratorGHI_SCNN(Sequence):

    'Generates data for Keras'
    def __init__(self, dataframe, img_folder, batch_size=128, dim=(128, 128, 3), channel_IMG = 1, shuffle=False, iftest=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_folder = img_folder
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        # seed = np.random.randint(0, 1000)
        # flip = RandomFlip("vertical", seed=seed) # or "horizontal", "vertical"
        # rotate = RandomRotation(0.2, seed=seed)
        
        
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            irr = vector_row.Irr
            for i_IMG in range(len(img_row)): # stack images in channels
                img = cv2.imread(os.path.join(self.img_folder, img_row[i_IMG]), cv2.IMREAD_GRAYSCALE)
                # img = tf.io.read_file(os.path.join(self.img_folder, img_row[i_IMG]))
                # img = tf.image.decode_jpeg(img, channels=1)
                # img = img.numpy().reshape((128, 128, 1)).astype("uint8")
                ghi = irr[i_IMG]
                w = int(np.floor(ghi/255.0))     # number of white pixels (val=255)
                last = np.mod(ghi, 255.0)       # value of the last pixel
                img[0][:w] = 255.0
                img[0][w] = last

                X[n_index, :, :, i_IMG] = img.reshape((128, 128)).astype("uint8") / 255.0
                    
            y[n_index] = value_row

            return X, y


def check_summer_time(date: datetime) -> bool:
    
    if date > datetime(date.year, 3, 31) and date < datetime(date.year, 10, 31):
        return True
    
    return False

def generate_dataframe(GHI_PATH: str, OUTPUT_PATH: str, FORECAST_HORIZON: int, SEQUENCE_HORIZON = None, CLEAR_SKY_MODEL='simplified_solis') -> pd.DataFrame:
    """Example of the dataframe generation funcion

    Args:
        GHI_PATH (str): path to weather station files. It is necessary that they contain GHI information.
        FORECAST_HORIZON (int): Forecast horizon in minutes, e.g., 30
        SEQUENCE_HORIZON (_type_, optional): Distance between images in sequence used to predict. Defaults to half of FORECAST_HORIZON.
        CLEAR_SKY_MODEL (str, optional): CS model type. Defaults to 'simplified_solis'.

    Returns:
        pd.DataFrame: Concatenated df with necessary columns to be fed into DataGenerator_SCNN()
    """
    
    df_data = pd.DataFrame()

    # Create a df for the model runs. The most important variables present in the final df are 'Image', 'Irr', 'Target' 
    # 'Image' - contains a list of names for the image sequence
    # 'Irr' - contains a list of measured irradiances for the instances of the Image
    # 'Terget' - contains the target ghi for the prediction in the specified horizon

    EPSILON = 1e-6
    if SEQUENCE_HORIZON == None:
        SEQUENCE_HORIZON = FORECAST_HORIZON // 2

    for day in os.listdir(GHI_PATH):

        day_date = datetime.strptime(day, '%Y%m%d.csv')

        # Load data
        csv_path = os.path.join(GHI_PATH, day)
        df_test = pd.read_csv(csv_path)

        # Change the str to datetime format
        df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d %H:%M:%S')
        
        # Get sun position and the clear sky parameters based on the chosen sky model
        times = pd.date_range(df_test['date'].min(), df_test['date'].max(), freq='1T')
        loc = location.Location(latitude = 45.5, longitude = 9.15)
        clear_sky = loc.get_clearsky(times, model=CLEAR_SKY_MODEL)
        solpos = solarposition.get_solarposition(times, latitude=45.5, longitude=9.15)
        
        # Merge the data into a single day df
        df_sun = clear_sky.merge(solpos[['elevation', 'azimuth']], left_index=True, right_index=True)
        df_day = df_test.merge(df_sun, left_on='date', right_index=True)
        
        filter_artifacts = (df_day['ghi1'] > 1500)
        df_day.loc[filter_artifacts, 'ghi1'] = np.nan
        df_day.ghi1.fillna(method='ffill', inplace=True)

        # Add the corresponding image name
        
        
        # In winter time, the data is shifted by 1 hour
        # The algorithm should take the time shift into account by taking an earlier image (-1 hour)
        if check_summer_time(day_date):
            img_list = [datetime.strftime(date, '%Y%m%d\%Y%m%d%H%M%S')+'.jpg' for date in df_day['date']]
        else:
            img_list = [datetime.strftime(date+timedelta(hours=-1), '%Y%m%d\%Y%m%d%H%M%S')+'.jpg' for date in df_day['date']]
        
        df_day['img'] = img_list
        df_day['img'] = [item if os.path.exists(os.path.join(OUTPUT_PATH, item)) else None for item in df_day.img]
        df_day['CSI'] = df_day.ghi1.values / (df_day.ghi.values + EPSILON)

        # Add the desired 3 columns
        Image = np.stack((df_day.img.shift(2 * SEQUENCE_HORIZON).values, df_day.img.shift(SEQUENCE_HORIZON).values, df_day.img.values), axis = 1)
        Irr = np.stack((df_day.ghi1.shift(2 * SEQUENCE_HORIZON).values, df_day.ghi1.shift(SEQUENCE_HORIZON).values, df_day.ghi1.values), axis = 1)
        Target_GHIr = df_day.ghi1.shift(-FORECAST_HORIZON).values # Just ghi, the divitsion into CSI will be done later
        Target_CSI = df_day.CSI.shift(-FORECAST_HORIZON).values
        Target_GHICS = df_day.ghi.shift(-FORECAST_HORIZON).values

        Image_list = [item.tolist() for item in Image]
        Irr_list = [item.tolist() for item in Irr]

        df_day['Irr'] = Irr_list
        df_day['Image'] = Image_list
        df_day['Target_GHIr'] = Target_GHIr
        df_day['Target_CSI'] = Target_CSI
        df_day['Target_GHICS'] = Target_GHICS

        df_data = pd.concat((df_data, df_day), ignore_index=True)

    return df_data