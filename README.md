# Detection-aided liver lesion segmentation

Here we present the implementation in TensorFlow of this [work]() accepted in the Machine Learning 4 Health Workshop from NIPS. The original code is from [OSVOS](https://github.com/scaelles/OSVOS-TensorFlow).

![Architecture of the network](https://github.com/imatge-upc/medical-2017-liver/blob/master/img/architecture.png?raw=true)

 In this work we propose a method to segment the liver and its lesions from Computed Tomography (CT) scans using Convolutional Neural Networks (CNNs), that have proven good results in a variety of computer vision tasks, including medical imaging. The network that segments the lesions consists of a cascaded architecture, which first focuses on the region of the liver in order to segment the lesions on it. Moreover, we train a detector to localize the lesions, and mask the results of the segmentation network with the positive detections. The segmentation architecture is based on DRIU~\cite{maninis2016deep}, a Fully Convolutional Network (FCN) with side outputs that work on feature maps of different resolutions, to finally  benefit from the multi-scale information learned by different stages of the network. The main contribution of this work is the use of a detector to localize the lesions, which we show to be beneficial to remove false positives triggered by the segmentation network. 
 
 |  ![NIPS 2017 logo][logo-nips]   |  ![log ml4h][logo-ml4h]  | Paper accepted at [Machine Learning 4 Health Learning Workshop, NIPS 2017](https://ml4health.github.io/2017/index.html)   |
|:-:|:-:|---|    

[logo-nips]: https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/nips2017.png?raw=true "NIPS 2017"
[logo-ml4h]: https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/ml4h_smaller.png?raw=true "ML4H Workshop"

A joint collaboration between:

|![logo-eth]|![logo-bsc]|![logo-gpi]|
|:-:|:-:|:-:|
| [Eidgenössische Technische Hochschule Zürich][eth-zurich] | [Barcelona Supercomputing Center][bsc-web] | [UPC Image Processing Group][gpi-web] |

[eth-zurich]: http://www.vision.ee.ethz.ch/en/
[gpi-web]: https://imatge.upc.edu/web/ 
[bsc-web]: http://www.bsc.es 

[logo-eth]:https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/ethzurich_smaller.jpeg?raw=true "ETH Zürich"
[logo-bsc]:https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/bsc320x86.jpg?raw=true "Barcelona Supercomputing Center"
[logo-gpi]: https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/gpi320x70.png?raw=true "UPC Image Processing Group"
[logo-severo]: https://github.com/imatge-upc/medical-2017-liver/blob/master/logos/severo_ochoa.png?raw=true "Severo Ochoa"

| ![Míriam Bellver][bellver-photo]  | ![Kevis-Kokitsi Maninis][maninis-photo]  | ![Jordi Pont-Tuset][pont-photo]  | ![Xavier Giró i Nieto][giro-photo]  | ![Jordi Torres][torres-photo]  |![Luc Van Gool][gool-photo]|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Míriam Bellver][bellver-web] | [Kevis-Kokitsi Maninis][maninis-web]  | [Jordi Pont-Tuset][pont-web] | [Xavier Giro-i-Nieto][giro-web]  |  [Jordi Torres][torres-web] | [Luc Van Gool][gool-web]  |


[bellver-web]: https://www.bsc.es/bellver-bueno-miriam
[maninis-web]: http://www.vision.ee.ethz.ch/~kmaninis/
[pont-web]: http://jponttuset.cat/publications/
[giro-web]: https://imatge.upc.edu/web/people/xavier-giro
[torres-web]: http://www.jorditorres.org/
[gool-web]: http://www.vision.ee.ethz.ch/en/members/get_member.cgi?name=vangool&lang=en#

[bellver-photo]:  https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/MiriamBellver160x160.jpg?raw=true "Míriam Bellver"
[maninis-photo]: https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/KManinis160x160.jpg?raw=true "Kevis-Kokitsi Maninis"
[pont-photo]: https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/JordiPont160x160.jpg?raw=true "Jordi Pont-Tuset"
[giro-photo]: https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/XavierGiro160x160.jpg?raw=true "Xavier Giró-i-Nieto"
[torres-photo]: https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/JordiTorres160x160.jpg?raw=true "Jordi Torres"
[gool-photo]:  https://github.com/imatge-upc/medical-2017-liver/blob/master/authors/lucvangool160x160.jpg?raw=true  "Luc Van Gool"


## Code Instructions





### Setup

Virtualenv etc

If you want to test our models, download the different weights:

[Liver segmentation checkpoint]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/
[Lesion segmentation checkpoint]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/
[Lesion detection checkpoint]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/

If you want to learn them, use the following checkpoints:

[VGG-16 weights]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/
[Resnet-50 weights weights]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/


##### Database

This code was developed to participate in the [Liver lesion segmentation challenge (LiTS)](https://competitions.codalab.org/competitions/17094), but can be used for other segmentation tasks also. The LiTS database consists on 130 CT scans for training and 70 CT scans for testing. In the training set, we did our own partition, being CT scans with ids lower than 105 being used for training and the remanining for testing. They are compressed in a nifti format. 

Our code expects that the folder that contains the database, has the following structure. It should have the following folders insise:

images_volumes: inside there should be a folder for each volume. Inside each of this folders, there are all the .mat files (each matfile corresponds to one slice). The preprocessing is clipping the values outside the range (-150,250) and doing max-min normalization.
liver_seg: the same structure as the previous, but with .png for each mask of the liver (or any other structure you decide)
item_seg: the same structure as the previous, but with .png for each mask of the lesion (or any other structure you decide)

Database/108/images_volumes/100.mat
Database/108/liver_seg/100.png
Database/108/item_seg/100.png

We provide a file in matlab to convert these nifti files into this same structure, if you do. In our case we used [this](https://ch.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image) matlab library. You can use whatever library you decide.

```bash
cd /utils/matlab_files
matlab process preprocess_liver.m
```

### Liver segmentation

**1. Train the liver model**

In seg_liver_train.py you should indicate a dataset list file. An example is inside seg_DatasetList, training_volume_3.txt. Each line has

img1 seg_lesion1 seg_liver1 img2 seg_lesion2 seg_liver2 img3 seg_lesion3 seg_liver3 

If you just have segmentations of the liver, then repeate seg_lesionX=seg_liverX. If you used the folder structure explained in the previous point, you can use the training and testing_volume_3.txt files.

```bash
python seg_liver_train.py
```

**2. Test the liver model**

A dataset list with the same format but with the test images is required here. If you don't have annotations, simply put a dummy annotation X.png. There is also an example in seg_DatasetList/testing_volume_3.txt. 

```bash
python seg_liver_test.py
```

### Lesion detection

This network samples locations around liver and detects whether they have a lesion or not. 

**1. Sample locations**

The first step is to sample these locations. We need a .txt with the following format:

img1 x1 x2 id

Example:

images_volumes/97/444 385.0 277.0 1

x1 and x2 are the coordinates of the upper-left vertex of the bounding box and id is the data augmentation option.  There are two options in this script. To sample locations for slices with ground truth or without. In the first case, two separate lists will be generated, one for positive locations (/w lesion) and another for negative locations (/wo lesion), in order to train the detector with balanced batches. These lists are already generated so you can use them, they are inside det_DatasetList (for instance, training_positive_det_patches_data_aug.txt for the positive patches of training set).

In case you want to generate other lists, use the following script:

```bash
cd utils/sampling
python sample_bbs.py
```

**2. Train lesion detector** 

Once you have sampled the positive and negative locations, or use the default lists, you can use the following command to train the detector.

```bash
python seg_lesion_train.py
```

**3. Test lesion detector** 

In order to test the detector, you can use the following command:

```bash
python seg_lesion_test.py
```

This will create in the folder detection_results/ a folder with the name of the id given to the experiment, and inside two .txt files, one with the hard results (considering a th of 0.5) and another with soft results with the prob predicted by the detector that a location is unhealthy.

### Lesion segmentation

**1. Crop slices around the liver**

In order to train the algorithm, first we crop the images around the liver region. If you don't have the cropped slices, you can use the script utils/crops_methods/compute_3D_bbs_from_gt_liver.py

```bash
cd utils/crops_methods
python compute_3D_bbs_from_gt_liver.py
```

This will generate three folders:

bb_liver_seg_alldatabase3_gt_nozoom_common_bb
bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb
bb_images_volumes_alldatabase3_gt_nozoom_common_bb


**2. Train the lesion model**

In order to train the algorithm that does not back propagate through pixels outside the liver, each line of the list.txt file should have the following format.

img1 seg_lesion1 seg_liver1 result_liver1 img2 seg_lesion2 seg_liver2 result_liver1 img3 seg_lesion3 seg_liver3 result_liver1 

An example list file is seg_DatasetList/training_lesion_commonbb_nobackprop_3.txt. If you used the folder structure proposed in the Database section, and you have named the folders of the cropped slices as proposed in the compute_3D_bbs_from_gt_liver.py file, you can use the seg_DatasetList/training_lesion_commonbb_nobackprop_3.txt and the one for testing. 

```bash
python seg_lesion_train.py
```

**3. Test the lesion model**

The command to test the network is the following:

```bash
python seg_lesion_test.py
```

In this case, observe that the script does 4 different steps
1. Do inference with the network
2. Return results to the original size (from cropped slices to 512x512)
3. Mask the results with the liver segmentation masks 
4. Check positive detections of lesions in the liver. Remove those false positive of the segmentation network using the detection results.  (You need the detection results, obtained in the next point!)

## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC. We also would like to thank Carlos Tripiana from the technical support team at the Barcelona Supercomputing center (BSC). 

| ![AlbertGil-photo]  | ![JosepPujal-photo]  | ![CarlosTripiana-photo]  |
|:-:|:-:|:-:|
| [Albert Gil](https://imatge.upc.edu/web/people/albert-gil-moreno)  |  [Josep Pujal](https://imatge.upc.edu/web/people/josep-pujal) | [Carlos Tripiana](https://www.bsc.es/tripiana-carlos/) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"
[CarlosTripiana-photo]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/authors/carlos160x160.jpeg?raw=true "Carlos Tripiana"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal
[CarlosTripiana-web]: https://www.bsc.es/tripiana-carlos/

|   |   |
|:--|:-:|
| This work has been supported by the [grant SEV2015-0493 of the Severo Ochoa Program](https://www.bsc.es/es/severo-ochoa/presentaci%C3%B3n) awarded by Spanish Government, project TIN2015-65316 by the Spanish Ministry of Science and Innovation contracts 2014-SGR-1051 by Generalitat de Catalunya | ![logo-severo] |
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work at the UPC, and the BSC/UPC NVIDIA GPU Center of Excellence. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the project [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 


[logo-nvidia]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/logos/excellence_center.png?raw=true  "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"


## Publication

Our workshop paper is available on [arXiv](), and related slides [here](https://www.slideshare.net/xavigiro/detectionaided-liver-lesion-segmentation-using-deep-learning).

Please cite with the following Bibtex code:

````
@InProceedings{Bellver_2016_NIPSWS,
author = {Bellver, Miriam and Giro-i-Nieto, Xavier and Marques, Ferran and Torres, Jordi},
title = {Hierarchical Object Detection with Deep Reinforcement Learning},
booktitle = {Deep Reinforcement Learning Workshop, NIPS},
month = {December},
year = {2016}
}
````

You may also want to refer to our publication with the more human-friendly Chicago style:

*Miriam Bellver, Xavier Giro-i-Nieto, Ferran Marques, and Jordi Torres. "Hierarchical Object Detection with Deep Reinforcement Learning." In Deep Reinforcement Learning Workshop (NIPS). 2016.*


## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/detection-2016-nipsws/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:miriam.bellver@bsc.es>.
