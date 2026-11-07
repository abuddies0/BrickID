#
# File: main.py
#
# Description:  Trains a CNN model to recognize bricks
#
###################################################################

# import sys
# import tensorflow as tf

# print("Python executable:", sys.executable)
# print("TensorFlow version:", tf.__version__)
# print("TensorFlow location:", tf.__file__)
# print("Build info:", tf.sysconfig.get_build_info())
# print("GPUs:", tf.config.list_physical_devices("GPU"))


import tensorflow as tf

with tf.device("/GPU:0"):
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    c = tf.matmul(a, b)

print(c.device)