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
import det_lesion as detection
from dataset.dataset_det import Dataset

gpu_id = 0

task_name = 'det_lesion_ck'

database_root = os.path.join(root_folder, 'LiTS_database')
logs_path = os.path.join(root_folder, 'train_files', task_name, 'networks')
result_root = os.path.join(root_folder, 'detection_results/')

model_name = os.path.join(logs_path, "det_lesion.ckpt")

val_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_positive_det_patches.txt')
val_file_neg = os.path.join(root_folder, 'det_DatasetList', 'testing_negative_det_patches.txt')

dataset = Dataset(None, None, val_file_pos, val_file_neg, None, database_root, store_memory=False)

result_path = os.path.join(result_root, task_name)
checkpoint_path = model_name
detection.validate(dataset, checkpoint_path, result_path, number_slices=1)

"""For testing dataset without labels
# test_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_det_patches.txt')
# dataset = Dataset(None, None, test_file_pos, None, None, database_root, store_memory=False)
# detection.test(dataset, checkpoint_path, result_path, number_slices=1, volume=False)
"""