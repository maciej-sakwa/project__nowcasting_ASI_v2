import psutil
import tensorflow as tf

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

