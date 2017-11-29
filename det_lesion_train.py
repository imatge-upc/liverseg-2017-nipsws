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
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import det_lesion as detection
from dataset.dataset_det_data_aug import Dataset

gpu_id = 0

# Training parameters
batch_size = 64
iter_mean_grad = 1
max_training_iters = 5000
save_step = 200
display_step = 2
learning_rate = 0.01

task_name = 'det_lesion'

database_root = os.path.join(root_folder, 'LiTS_database')
logs_path = os.path.join(root_folder, 'train_files', task_name, 'networks')
resnet_ckpt = os.path.join(root_folder, 'train_files', 'resnet_v1_50.ckpt')

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
                        resume_training=False)
