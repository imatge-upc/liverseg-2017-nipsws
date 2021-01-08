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
import seg_liver as segmentation
from dataset.dataset_seg import Dataset
from config import Config

number_slices = 3

task_name = 'seg_liver_ck'

### config constants ###
config = Config()
database_root = config.database_root
logs_path = config.get_log(task_name)
result_root = config.get_result_root('results')
root_folder = config.root_folder
###

model_name = os.path.join(logs_path, "seg_liver.ckpt")

test_file = os.path.join(root_folder, 'seg_DatasetList/testing_volume_3.txt')

dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

result_path = os.path.join(result_root, task_name)
checkpoint_path = model_name
segmentation.test(dataset, checkpoint_path, result_path, number_slices)
