import numpy as np
from scipy import misc
import os


def crop(base_root, input_config='lesion/',
         crops_list='./crops_list/crops_LiTS_gt.txt'):

    crops_list = base_root + '/utils/crops_list/' + crops_list
    input_results_path = base_root + '/results/' + input_config
    output_results_path = base_root + '/results/out_' + input_config

    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()

    for i in range(len(crops_lines)):
            result = crops_lines[i].split(' ')

            if len(result) > 2:
                id_img, bool_zoom, mina, maxa, minb, maxb = result
                mina = int(mina)
                maxa = int(maxa)
                minb = int(minb)
                maxb = int(maxb.split('\n')[0])
            else:
                id_img, bool_zoom = result

            if int(id_img.split('/')[-2]) > 104:
                if not os.path.exists(os.path.join(output_results_path, id_img.split('/')[0])):
                    os.makedirs(os.path.join(output_results_path, id_img.split('/')[0]))

                mask = np.zeros([512, 512])
                if bool_zoom == '1':
                    zoomed_mask = misc.imread(os.path.join(input_results_path, id_img + '.png'))
                    mask[mina:maxa, minb:maxb] = zoomed_mask
                misc.imsave(os.path.join(output_results_path, id_img + '.png'), mask*255)

        
        
