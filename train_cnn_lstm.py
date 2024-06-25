import os, yaml, mlflow

import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
from pathlib import Path

from src.conv_lstm import conv_lstm
from src.model_utils import *
from src.weather_data_preprocessing import *
from utils.dotdict import DotDict


#TF data functions
def parse_sequence(row):
    
    features_list = []
    
    for i in range(8):
        img = tf.io.read_file(row[i])
        img = tf.image.decode_jpeg(img, channels=1)
        features_list.append(img)
        
    features = tf.stack(features_list, axis=0)
    label = tf.strings.to_number(row[-1], out_type=tf.dtypes.float32)
    
    return features, label

def compile_dataset_from_dataframe(df, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(df.values)
    ds = ds.map(parse_sequence)
    ds = ds.batch(batch_size)

    return ds.prefetch(1)

# main
def setup(config):

    ### SETUP ###

    IMAGE_PATH  = Path(config.PATHS.images)
    METEO_PATH  = Path(config.PATHS.sensor)
    OUT_PATH    = Path(config.PATHS.out_final)

    ELEVATION   = config.PARAMS.elevation_threshold
    EPSILON     = config.PARAMS.epsilon

    # Create the filenames dataframe
    df = get_sequence_dataframe(IMAGE_PATH, n_sequece=8)

    # Load the old file with 15 min horizon
    df_meteo = pd.read_parquet(METEO_PATH)
    
    # Index cleaning
    df_meteo.index = pd.to_datetime(df_meteo['date'])
    df_meteo.index = df_meteo.index.tz_convert(None)
    df_meteo.index.name = None

    # Get variables of interest
    df_meteo_targets = df_meteo[['Target_GHIr', 'Target_GHICS']].copy()

    # Scale the data
    df_load_targets_max = df_meteo_targets / df_meteo_targets.max()

    # Add the elevation
    df_load_targets_max = pd.concat([df_load_targets_max, df_meteo[['elevation']].copy()], axis=1)

    # Add the CSI
    df_load_targets_max['CSI'] = df_load_targets_max['Target_GHIr'].values / (df_load_targets_max['Target_GHICS'].values + EPSILON)

    # output the data
    df_load_final = df_load_targets_max[df_load_targets_max.index.isin(df.index)]

    # Create final dataframe
    df_final = pd.concat([df, df_load_final[['CSI', 'elevation']]], axis=1)

    # Filter the data by elevation
    df_final.dropna(inplace=True)
    df_final = df_final[df_final.elevation > ELEVATION].copy()
    df_final.drop(columns=['elevation'], inplace=True)

    # Standardize the label
    df_final['CSI'] = ((df_final['CSI'] - df_final['CSI'].mean()) / df_final['CSI'].std())

    # Homogenise the data type
    df_final = convert_to_str(df_final)

    # Remove index
    df_final.reset_index(inplace=True, drop='index')
    
    # Save df
    df_final.to_parquet(OUT_PATH)

def train(config):

    ### TRAIN ###

    tf.keras.backend.clear_session()
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    
    
    mlflow.set_tracking_uri(config.MLFLOW.tracking_uri)
    mlflow.set_experiment(config.MLFLOW.experiment_name)


    # Load config
    N_EPOCHS    = config.RUN.epochs
    PATIENCE    = config.RUN.patience
    LR_PATIENCE = config.RUN.lr_patience
    LR_START    = config.RUN.lr_start
    LOSS        = config.RUN.loss
    BETA_1      = config.RUN.beta_1
    BETA_2      = config.RUN.beta_2

    BATCH_SIZE  = config.PARAMS.batch_size
    INPUT_DIM   = config.PARAMS.dim
    TRAIN_SPLIT = config.PARAMS.train_split
    VALID_SPLIT = config.PARAMS.val_split

    N_RESIDUAL  = config.MODEL.residual_layers
    N_FILTERS   = config.MODEL.residual_filters
    N_LSTM_CELL = config.MODEL.lstm_units

    OUT_PATH    = config.PATHS.out_final

    # Prepare ds
    df = pd.read_parquet(OUT_PATH)
    len_data = len(df)

    df_train = df[:int(TRAIN_SPLIT*len_data)]
    df_val = df[int(TRAIN_SPLIT*len_data):int((TRAIN_SPLIT+VALID_SPLIT)*len_data)]
    df_test = df[int((TRAIN_SPLIT+VALID_SPLIT)*len_data):]

    ds_train = compile_dataset_from_dataframe(df_train)
    ds_val = compile_dataset_from_dataframe(df_val)

    # Define model
    model = conv_lstm(input_shape=INPUT_DIM, residual_filters=N_FILTERS, residual_layers=N_RESIDUAL, lstm_filters=N_LSTM_CELL)


    RUN_ID      = config.MODEL.run_id
    MODEL_TYPE  = config.MODEL.type
    RUN_NAME = f'{RUN_ID:03d}_{MODEL_TYPE}, lr: {LR_START}'

    # Make checkpoint path
    CHECKPOINT_PATH = Path(config.PATHS.checkpoint)
    if CHECKPOINT_PATH.exists() is False: 
        os.makedirs(CHECKPOINT_PATH)
    CHECKPOINT_PATH_RUN = os.path.join(CHECKPOINT_PATH, RUN_NAME.split(',')[0] + '.h5')


    with mlflow.start_run(run_name=RUN_NAME):
        
        mlflow.tensorflow.autolog()
        mlflow.set_tag("mlflow.note.content", config['MLFLOW']['note'])

        
        
        # Callbacks
        callbacks_list = []
       
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filepath=CHECKPOINT_PATH_RUN,
            verbose = 0, save_best_only = True)
        
        cb_earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            verbose=0)
        
        cb_memory = MemoryCallback()
        
        callbacks_list.append(cb_checkpoint)
        callbacks_list.append(cb_earlystop)
        callbacks_list.append(cb_memory)


        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_START, beta_1=BETA_1, beta_2=BETA_2)
        
        model.compile(optimizer=optimizer, loss=LOSS)
        mlflow.log_param("trainable_params", model.count_params())

        history = model.fit(ds_train, epochs=N_EPOCHS, validation_data = ds_val, callbacks=callbacks_list)
        mlflow.log_param("trainable_params", model.count_params())

    ### TEST ###

def main():

    # Open config
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        config = DotDict(config)

    setup(config)
    train(config)




if __name__ == "__main__":
    main()



