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
import det_lesion as detection
from dataset.dataset_det_data_aug import Dataset
from config import Config

gpu_id = 0

# Training parameters
batch_size = 64
iter_mean_grad = 1
# max_training_iters = 5000
max_training_iters = 500

save_step = 200
display_step = 2
learning_rate = 0.01

task_name = 'det_lesion'

### config constants ###
config = Config()
database_root = config.database_root
logs_path = config.get_log(task_name)
root_folder = config.root_folder
resnet_ckpt = config.resnet_ckpt
###

train_file_pos = os.path.join(root_folder, 'det_DatasetList', 'training_positive_det_patches_data_aug.txt')
train_file_neg = os.path.join(root_folder, 'det_DatasetList', 'training_negative_det_patches_data_aug.txt')
val_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_positive_det_patches_data_aug.txt')
val_file_neg = os.path.join(root_folder, 'det_DatasetList', 'testing_negative_det_patches_data_aug.txt')

dataset = Dataset(train_file_pos, train_file_neg, val_file_pos, val_file_neg, None, None, database_root=database_root,
                  store_memory=False)

# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        detection.train(dataset, resnet_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step,
                        global_step, iter_mean_grad=iter_mean_grad, batch_size=batch_size, finetune=0,
                        resume_training=False) # Make true to resume
