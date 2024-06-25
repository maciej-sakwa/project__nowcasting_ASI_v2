import os

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from pvlib import location, solarposition




"""
Data processing module. Contains definition of the classes and functions used to stack data.
V.1.0.0 MS 09/10/23

Functions:
|- generate_dataframe(GHI_PATH, FORECAST_HORIZON, SEQUENCE_HORIZON = None, CLEAR_SKY_MODEL='simplified_solis') - Example of the dataframe generation funcion
"""

# Dataframe functions 
def get_date_range(image_names):
    
    date_min = min(image_names)
    date_max = max(image_names)

    date_min_date = datetime.strptime(date_min, '%Y%m%d%H%M%S')
    date_max_date = datetime.strptime(date_max, '%Y%m%d%H%M%S')

    date_times = pd.date_range(date_min_date, date_max_date, freq='1min')
    
    return date_times

def get_sequence_dataframe(image_path, n_sequece=8):
    
    # Get all the images paths
    t_0 = [item for item in image_path.glob('*/*')]
    date = [item.stem for item in t_0]
    
    # Get the date range
    date_times = get_date_range(date)
    
    # Create a dataframe with the images paths
    file_names = [(str(image_path) + datetime.strftime(item, '/%Y%m%d/%Y%m%d%H%M%S') + '.jpg') for item in date_times]
    img_exists = [os.path.exists(item) for item in file_names]
    df_images = pd.DataFrame({'dates': date_times, 't_0': file_names, 'e_0': img_exists})
    
    # Create the sequence dataframe by shifting the image paths
    for i in range(1, n_sequece, 1):
        df_images[f't_{i}'] = df_images['t_0'].shift(i)
        df_images[f'e_{i}'] = df_images['e_0'].shift(i)

    # Filter the dataframe to remove the rows with missing values
    filter = [all(item) for item in df_images[['e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6', 'e_7']].values]
    df = df_images[filter].dropna()
    
    # Create the sequence dataframe
    df.index = df['dates']
    df.index.name = None
    df = df.drop(columns=['dates', 'e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6', 'e_7'])
    
    return df
    
def convert_to_str(df):
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


def check_summer_time(date: datetime) -> bool:
    
    if date > datetime(date.year, 3, 31) and date < datetime(date.year, 10, 31):
        return True
    
    return False

def generate_weather_dataframe(GHI_PATH: Path, FORECAST_HORIZON: int, CLEAR_SKY_MODEL='simplified_solis') -> pd.DataFrame:
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

    EPSILON = 1e-6


    for day in GHI_PATH.glob('*'):

        # Load data
        df_test = pd.read_csv(GHI_PATH / day)

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
        
        # Filter recording artifacts
        filter_artifacts = (df_day['ghi1'] > 1500)
        df_day.loc[filter_artifacts, 'ghi1'] = np.nan
        df_day['ghi1'] = df_day['ghi1'].interpolate(method='time')

        # Define CSI
        df_day['CSI'] = df_day.ghi1.values / (df_day.ghi.values + EPSILON)

        # Add the desired 3 columns
        df_day['Target_GHIr'] = df_day.ghi1.shift(-FORECAST_HORIZON).values
        df_day['Target_CSI'] = df_day.CSI.shift(-FORECAST_HORIZON).values
        df_day['Target_GHICS'] = df_day.ghi.shift(-FORECAST_HORIZON).values

        df_data = pd.concat((df_data, df_day), ignore_index=True)

    return df_data