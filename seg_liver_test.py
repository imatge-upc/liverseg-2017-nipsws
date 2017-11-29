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
import numpy as np
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import seg_liver as segmentation
from dataset.dataset_seg import Dataset

number_slices = 3

task_name = 'seg_liver_ck'

database_root = os.path.join(root_folder, 'LiTS_database')
logs_path = os.path.join(root_folder, 'train_files', task_name, 'networks')
result_root = os.path.join(root_folder, 'results')
model_name = os.path.join(logs_path, "seg_liver.ckpt")

test_file = os.path.join(root_folder, 'seg_DatasetList/testing_volume_3.txt')

dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

result_path = os.path.join(result_root, task_name)
checkpoint_path = model_name
segmentation.test(dataset, checkpoint_path, result_path, number_slices)
