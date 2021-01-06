import numpy as np
from scipy import misc
import os
import glob
import math
import scipy.io


MIN_AREA_SIZE = 512.0*512.0

## this file is generated at the end 
crops_list_name = 'crops_LiTS_gt_2.txt'

database_root = '../../predict_database/'

utils_path = '../crops_list/'
results_path = '../../results/'
images_path = os.path.join(database_root, 'images_volumes')
labels_path = os.path.join(database_root,  'item_seg/')
labels_liver_path = os.path.join(database_root,  'liver_seg/')

output_images_path_bb = os.path.join(database_root, 'bb_images_volumes_alldatabase3_gt_nozoom_common_bb')
output_labels_path_bb = os.path.join(database_root,  'bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb')
output_labels_liver_path_bb = os.path.join(database_root,  'bb_liver_seg_alldatabase3_gt_nozoom_common_bb')

liver_results = os.path.join(database_root, 'seg_liver_ck/')
output_liver_results_path_bb = os.path.join(database_root, 'liver_results/')

# This script computes the bounding boxes around the liver from the ground truth, computing
# a single 3D bb for all the volume.

bb_paths = [output_labels_path_bb, output_images_path_bb, output_labels_liver_path_bb, output_liver_results_path_bb]

for bb_path in bb_paths:
    if not os.path.exists(bb_path):
        os.makedirs(bb_path)

## If no labels, the masks_folder should contain the results of liver segmentation
# masks_folders = os.listdir(results_path + 'liver_seg/')
masks_folders = os.listdir(labels_liver_path) # liver seg
sorted_mask_folder = sorted(masks_folders, key=lambda x: int(x))

crops_file = open(os.path.join(utils_path, crops_list_name), 'w')
aux = 0

sort_by_path = lambda x: int(os.path.splitext(os.path.basename(x))[0])

for i in range(len(masks_folders)):
    if not masks_folders[i].startswith(('.', '\t')):
        dir_name = masks_folders[i]
        ## If no labels, the masks_folder, should contain the results of liver segmentation
        masks_of_volume = glob.glob(labels_liver_path + dir_name + '/*.png')
        file_names = (sorted(masks_of_volume, key=sort_by_path))
        depth_of_volume = len(masks_of_volume)

    bb_paths = [output_labels_path_bb, output_images_path_bb, output_labels_liver_path_bb, output_liver_results_path_bb]
    
    for bb_path in bb_paths:
        if not os.path.exists(os.path.join(bb_path, dir_name)):
            os.makedirs(os.path.join(bb_path, dir_name))

        
    total_maxa = 0
    total_mina = 10000000
    
    total_maxb = 0
    total_minb = 10000000
        
    for j in range(0, depth_of_volume):
        img = misc.imread(file_names[j])
        img = img/255.0
        img[np.where(img > 0.5)] = 1
        img[np.where(img < 0.5)] = 0
        a, b = np.where(img == 1)
        
        if len(a) > 0:

            maxa = np.max(a)
            maxb = np.max(b)
            mina = np.min(a)
            minb = np.min(b)
            
            if maxa > total_maxa:
                total_maxa = maxa
            if maxb > total_maxb:
                total_maxb = maxb
            if mina < total_mina:
                total_mina = mina
            if minb < total_minb:
                total_minb = minb
            
    for j in range(0, depth_of_volume):
        img = misc.imread(file_names[j])
        img = img/255.0
        img[np.where(img > 0.5)] = 1
        img[np.where(img < 0.5)] = 0

        a, b = np.where(img == 1)
        
        if len(a) > 0:

            new_img = img[total_mina:total_maxa, total_minb:total_maxb]

        # elements of txt line
        current_file_path = file_names[j].split('.png')[0]
        current_file = os.path.basename(os.path.splitext(current_file_path)[0])
        print("current file ->",current_file)
        png = current_file.split('/')[-1] + '.png'
        mat = current_file.split('/')[-1] + '.mat'
        liver_seg = current_file.split('liver_seg/')[-1]

        if len(np.where(img == 1)[0]) > 500:

            # constants
            area = 1
            zoom = math.sqrt(MIN_AREA_SIZE/area)
            aux = 1

            

            # write to crops txt file
            line = ' '.join([str(x) for x in [liver_seg, aux, total_mina, total_maxa, total_minb, total_maxb]])
            crops_file.write(line + '\n')

            ######### apply 3Dbb to files ##########
            print("images_path",images_path)
            print("dir_name", dir_name)
            print("mat", mat)
            print("png", png)
            # .mat
            original_img = np.array(scipy.io.loadmat(os.path.join(images_path, dir_name, mat))['section'], dtype = np.float32)
            o_new = original_img[total_mina:total_maxa, total_minb:total_maxb]
            scipy.io.savemat(os.path.join(output_images_path_bb, dir_name, mat), mdict={'section': o_new})


            ### DEPRECATED: masked_original_img is never used
            masked_original_img = o_new
            masked_original_img[np.where(new_img == 0)] = 0
            ###
        
            # lesion png
            original_label = misc.imread(os.path.join(labels_path, dir_name, png))
            lbl_new = original_label[total_mina:total_maxa, total_minb:total_maxb]
            misc.imsave(os.path.join(output_labels_path_bb, dir_name, png), lbl_new)
            
            # liver png
            original_liver_label = misc.imread(os.path.join(labels_liver_path, dir_name, png))
            lbl_liver_new = original_liver_label[total_mina:total_maxa, total_minb:total_maxb]
            misc.imsave(os.path.join(output_labels_liver_path_bb, dir_name,  png), lbl_liver_new)

            # results png
            original_results_label = misc.imread(os.path.join(liver_results, dir_name, png))
            res_liver_new = original_results_label[total_mina:total_maxa, total_minb:total_maxb]
            misc.imsave(os.path.join(output_liver_results_path_bb, dir_name, png), res_liver_new)

        else:
            aux = 0
            crops_file.write(current_file.split('liver_seg/')[-1]  + ' ' + str(aux) + '\n')

crops_file.close()