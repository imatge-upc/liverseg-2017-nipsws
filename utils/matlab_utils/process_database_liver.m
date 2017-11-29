addpath('../../tools/nifti_library/');

%folder with .nii files 
niftis_path = '../../Database/media/nas/01_Datasets/CT/LITS/Training Batch 1/';

root_process_database = '../../LiTS_database/';
folder_volumes = strcat(root_process_database, 'images_volumes/');
folder_seg_liver = strcat(root_process_database, 'liver_seg/');
folder_seg_item = strcat(root_process_database, 'item_seg/');


if ~(exist(root_process_database, 'dir'))
    mkdir(root_process_database);
end
if ~(exist(folder_volumes, 'dir'))
    mkdir(folder_volumes);
end
if ~(exist(folder_seg_liver, 'dir'))
    mkdir(folder_seg_liver);
end
if ~(exist(folder_seg_item, 'dir'))
    mkdir(folder_seg_item);
end

files_dir = dir(niftis_path);
copy_files_dir = files_dir;
filenames = [];

list_file_names = [];

% just keep those files that belong the volumes or segmentations
for k = 1:(length(files_dir))
	if (files_dir(k).name(1) == 'v' || files_dir(k).name(1) == 's' )
        filenames = [filenames files_dir(k)];
	end
end


for l = 1:(length(filenames))
	name = filenames(l).name;
    if (name(1) == 'v')
        display(['Processing Volume'])
        path_file = strcat(niftis_path, name);
        index = strfind(name,'.nii');
        folder_volume = strcat(folder_volumes, name((8):index-1));
        volume = load_nii(path_file);
        imgs = volume.img;
        imgs(imgs<-150) = -150;
        imgs(imgs>250) = 250;
        imgs = single(imgs);
        img_volume = 255*(imgs - min(imgs(:)))/(max(imgs(:)) - min(imgs(:)));
        if ~(exist(folder_volume, 'dir'))
            mkdir(folder_volume);
        end
		for j=1:(size(img_volume,3))
			section = img_volume(:,:,j);
            filename_for_section = strcat(folder_volume, '/', num2str(j), '.mat');
            save(filename_for_section, 'section', '-v6');
        end
    elseif (name(1) == 's')
        display(['Processing Segmentation'])
        path_file = strcat(niftis_path, name);
        index = strfind(name,'.nii');
        folder_seg_item_num = strcat(folder_seg_item, name(14:index-1));
        folder_seg_liver_num = strcat(folder_seg_liver, name(14:index-1));
        segmentation = load_nii(path_file);
        img_seg = uint8(segmentation.img);
        img_seg_item = img_seg;
        img_seg_liver = img_seg;
        img_seg_item(img_seg_item == 1)=0;
        img_seg_item(img_seg_item == 2)=1;
        img_seg_liver(img_seg_liver == 2)=1;
        if ~(exist(folder_seg_item_num, 'dir'))
            mkdir(folder_seg_item_num);
        end
        if ~(exist(folder_seg_liver_num, 'dir'))
            mkdir(folder_seg_liver_num);
        end
        for j=1:(size(img_seg_item,3))
            item_seg_section = img_seg_item(:,:,j)*255;
            liver_seg_section = img_seg_liver(:,:,j)*255;
            filename_for_seg_item_section = strcat(folder_seg_item_num, '/', num2str(j), '.png');
            filename_for_seg_liver_section = strcat(folder_seg_liver_num, '/', num2str(j),  '.png');
            imwrite(item_seg_section, filename_for_seg_item_section);
            imwrite(liver_seg_section, filename_for_seg_liver_section);
        end
    end
end
