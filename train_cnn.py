import os
import mlflow
import yaml
import psutil

import pandas as pd
import tensorflow as tf

from datetime import datetime
from sklearn import model_selection, metrics

from src import sequence_img_generator, get_models


with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    
# tf_config = tf.ConfigProto()

MODEL_TYPE          = config['MODEL']['type']
FORECAST_HORIZON    = config['MODEL']['forecast_horizon']
SEQUENCE_HORIZON    = config['MODEL']['sequence_horizon']
ELEVATION_THRESHOLD = config['MODEL']['elevation_threshold']

INPUT_PATH  = config['PATHS']['input_folder']
OUTPUT_PATH = config['PATHS']['output_folder']


def preprocess_sensor_data(path):
    df_data = sequence_img_generator.generate_dataframe(
        config['PATHS']['weather_files'], 
        OUTPUT_PATH = OUTPUT_PATH, 
        FORECAST_HORIZON = FORECAST_HORIZON, 
        SEQUENCE_HORIZON = SEQUENCE_HORIZON)
    df_data.drop(columns=['humidity', 'temperature', 'dni', 'dhi', 'azimuth'], inplace=True)
    df_data.to_parquet(path)
    
    
def filter_sensor_data(df_data, elevation_threshold):
    
    # Define the Target column according to needs
    df_data['Target'] = df_data.Target_CSI

    # Remove the data with low elevation
    df_data_reduced = df_data[df_data.elevation > elevation_threshold].copy()
    df_data_reduced.dropna(inplace=True)

    # Move to the data definition function
    dates = [datetime.strftime(date, '%Y-%m-%d %H:%M') for date in df_data_reduced.date]
    df_data_reduced.index = pd.to_datetime(dates)
    df_data_reduced.drop(columns=['date'], inplace=True)

    # Remove the data with missing images
    filter_nones = [False if None in img_list else True for img_list in df_data_reduced.Image]
    df_data_reduced = df_data_reduced[filter_nones].copy()
    filter_no_img = [os.path.exists(os.path.join(OUTPUT_PATH, img_path)) for img_path in df_data_reduced.img]
    df_data_reduced = df_data_reduced[filter_no_img].copy()

    return df_data_reduced

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def exponential_schedule_func(lr0, s):
    def exponential_schedule(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_schedule

class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print("\n")
        print(f"memory: {psutil.virtual_memory()[3] / 1024 / 1024} MB")
        print(f"memory: {psutil.virtual_memory()[2]} %")
        
        
def main():
    
    data_path = fr'data\df_data_{FORECAST_HORIZON}_{SEQUENCE_HORIZON}.parquet.gzip'
    if os.path.exists(data_path):
        df_data = pd.read_parquet(data_path)
    else:
        print('Preprocessing sensor data')
        preprocess_sensor_data(data_path)
        df_data = pd.read_parquet(data_path)
        
        
        
    tf.keras.backend.clear_session()
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    
    
    mlflow.set_tracking_uri(config['MLFLOW']['tracking_uri'])
    mlflow.set_experiment(config['MLFLOW']['experiment_name'])
    
    
    # Filter the data
    df_data_reduced = filter_sensor_data(df_data, ELEVATION_THRESHOLD)
    
    # Train test split
    df_test = df_data_reduced.loc['2024-05-14':].copy()
    df_train_full = df_data_reduced.loc[:'2024-05-13'].copy()
    df_train, df_val = model_selection.train_test_split(df_train_full, train_size=config['MODEL']['split'], shuffle=True)

    # Input generators
    train_generator = sequence_img_generator.DataGeneratorGHI_SCNN(df_train, OUTPUT_PATH, **config['PARAMS']['train_params'])
    
    val_generator = sequence_img_generator.DataGeneratorGHI_SCNN(df_val, OUTPUT_PATH, **config['PARAMS']['val_params'])
    
    
    
    model = get_models.SCNN_small(input_shape=config['MODEL']['img_size'])
    
    # Load config

    BATCH_SIZE  = config['PARAMS']['train_params']['batch_size']
    N_EPOCHS    = config['MODEL']['epochs']
    PATIENCE    = config['MODEL']['patience']
    LR_PATIENCE = config['MODEL']['lr_patience']
    LR_START    = config['MODEL']['lr_start']
    LOSS        = config['MODEL']['loss']
    BETA_1      = config['MODEL']['beta_1']
    BETA_2      = config['MODEL']['beta_2']
    
    
    
    RUN_ID = config['MODEL']['run_id']
    RUN_NAME = f'{RUN_ID:03d}_{MODEL_TYPE}, lr: {LR_START}, loss: {LOSS}, img_shifted'

    CHECKPOINT_PATH = config['PATHS']['checkpoint']

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.tensorflow.autolog()
        mlflow.set_tag("mlflow.note.content", config['MLFLOW']['note'])
        
        callbacks_list = []
        
        # Checkpointing
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        CHECKPOINT_PATH_RUN = os.path.join(CHECKPOINT_PATH, RUN_NAME.split(',')[0] + '.h5')

        
        callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filepath=CHECKPOINT_PATH_RUN,
                verbose = 0, save_best_only = True))

        # Early stopping
        callbacks_list.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, verbose=0))
        # Learing rate reduction scheduler
        # callbacks_list.append(get_models.OneCycleScheduler(math.ceil(len(df_train) / BATCH_SIZE) * N_EPOCHS, max_rate = 0.0005))
        callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        callbacks_list.append(MemoryCallback())
                    
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LR_START, 
            beta_1=BETA_1, 
            beta_2=BETA_2, 
            amsgrad=False
            )

        model.compile(
            optimizer=optimizer, 
            loss=LOSS,
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        history = model.fit(
            train_generator,
            epochs=N_EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks_list                              
            )
            
        mlflow.log_param("trainable_params", model.count_params())

                
        # Test the model               
        test_generator = sequence_img_generator.DataGeneratorGHI_SCNN(df_test, OUTPUT_PATH, **config['PARAMS']['test_params'])

            # Test ghi
        y_test = model.predict(test_generator) * df_test.Target_GHICS.values.reshape(-1, 1)
        y_true = df_test.Target_GHIr.values
        y_pers = df_test.ghi1.values
                                   
        mae_test = metrics.mean_squared_error(y_true, y_test)
        mae_per = metrics.mean_squared_error(y_true, y_pers)
                    
        FS = 1 - mae_test / mae_per
                
        print(f"model_params: {model.count_params()}")
        print(f"mae_test: {mae_test}")
        print(f"mae_pers {mae_per}")
        print(f"FS: {FS}")
                
        mlflow.log_metric(f"mae_test", mae_test)
        mlflow.log_metric(f"mae_pers", mae_per)
        mlflow.log_metric(f"FS", FS)
                
        mlflow.tensorflow.log_model(model, RUN_NAME.split(',')[0])
        mlflow.end_run()
        
    
if __name__ == '__main__':
    main()