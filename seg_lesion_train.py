"""
Original code from OSVOS (https://github.com/scaelles/OSVOS-TensorFlow)
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Modified code for liver and lesion segmentation:
Miriam Bellver (miriam.bellver@bsc.es)
"""

import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
import seg_lesion as segmentation
from dataset.dataset_seg_BPliver import Dataset
from config import Config

gpu_id = 0
number_slices = 3

# Training parameters
batch_size = 1
iter_mean_grad = 10
max_training_iters_1 = 15000
max_training_iters_2 = 30000
max_training_iters_3 = 50000
save_step = 2000
display_step = 2
ini_learning_rate = 1e-8
boundaries = [10000, 15000, 25000, 30000, 40000]
values = [ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate,
          ini_learning_rate * 0.1]

task_name = 'seg_lesion'

### config constants ###
config = Config()
database_root = config.database_root
logs_path = config.get_log(task_name)
root_folder = config.root_folder
imagenet_ckpt = config.imagenet_ckpt
###

train_file = os.path.join(root_folder, 'seg_DatasetList', 'training_lesion_commonbb_nobackprop_3.txt')
val_file = os.path.join(root_folder, 'seg_DatasetList', 'testing_lesion_commonbb_nobackprop_3.txt')

dataset = Dataset(train_file, None, val_file, database_root, number_slices, store_memory=False)

# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        segmentation.train_seg(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                           display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                           batch_size=batch_size, task_id=1, resume_training=False)

with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        segmentation.train_seg(dataset, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                           display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                           batch_size=batch_size, task_id=1, resume_training=True)

with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        segmentation.train_seg(dataset, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                           display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                           batch_size=batch_size, task_id=1, resume_training=True)
