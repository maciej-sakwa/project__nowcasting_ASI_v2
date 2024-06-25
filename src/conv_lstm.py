import tensorflow as tf


# Custom reshape layer (for some reason the tf.keras.layer.Reshape does not work)
class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        self.shape = target_shape
        super().__init__()

    def call(self, x):
        return tf.reshape(x, shape=self.shape)


# Residual block

def residual_block(x, filters, kernel_size=3, stride=1):
    
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def conv_lstm(input_shape, residual_filters, residual_layers, lstm_filters, kernel_size=3):
    
    # input shape is (batch, time, height, width, channels)
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    # x = tf.reshape(inputs, (-1, input_shape[1], input_shape[2], 1))
    #x = tf.keras.layers.Reshape((-1, input_shape[1], input_shape[2], 1))(inputs)
    x = ReshapeLayer((-1, input_shape[1], input_shape[2], 1))(inputs)
    
    # Convolutional preprocessing layers
    x = tf.keras.layers.Conv2D(residual_filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Convolutional residual layers for individual image processing
    for _ in range(residual_layers):
        x = residual_block(x, residual_filters, kernel_size)
    
    
    # Downsample to (32, 32, 1)
    x = tf.keras.layers.Conv2D(1, kernel_size, strides=4, padding='same')(x) # Downsample to (32, 32, 1)

    # Reshape for sequence processing (is this shape correct?)
    x = ReshapeLayer((-1, input_shape[0], input_shape[1]//4, input_shape[2]//4, 1))(x)
    
    # Convolutional LSTM layers for sequence processing
    x = tf.keras.layers.ConvLSTM2D(lstm_filters, kernel_size, padding='same', return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(lstm_filters, kernel_size, padding='same', return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(lstm_filters, kernel_size, padding='same', return_sequences=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Output layer
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs, x)
    
    return model