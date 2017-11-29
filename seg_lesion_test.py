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
import seg_lesion as segmentation
from dataset.dataset_seg import Dataset
import utils.crop_to_image
import utils.mask_with_liver
import utils.det_filter

gpu_id = 0
number_slices = 3

crops_list = 'crops_LiTS_gt.txt'
det_results_list = 'detection_lesion_example'
task_name = 'seg_lesion_ck'


database_root = os.path.join(root_folder, 'LiTS_database')
liver_results_path = os.path.join(database_root, 'out_liver_results')
logs_path = os.path.join(root_folder, 'train_files', task_name, 'networks')
result_root = os.path.join(root_folder, 'results')
model_name = os.path.join(logs_path, "seg_lesion.ckpt")

test_file = os.path.join(root_folder, 'seg_DatasetList/testing_volume_3_crops.txt')

dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

result_path = os.path.join(result_root, task_name)
checkpoint_path = model_name
segmentation.test(dataset, checkpoint_path, result_path, number_slices)
utils.crop_to_image.crop(base_root=root_folder, input_config=task_name, crops_list=crops_list)
utils.mask_with_liver.mask(base_root=root_folder, labels_path=liver_results_path, input_config='out_' + task_name, th=0.5)
utils.det_filter.filter(base_root=root_folder, crops_list=crops_list, input_config='masked_out_' + task_name,
                        results_list=det_results_list, th=0.33)



