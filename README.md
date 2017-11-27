# Detection-aided liver lesion segmentation

|  ![NIPS 2016 logo][logo-nips] | Paper accepted at [Machine Learning 4 Health Learning Workshop, NIPS 2017](https://ml4health.github.io/2017/index.html)   |
|:-:|---|


## Summary

 In this work we propose a method to segment the liver and its lesions from Computed Tomography (CT) scans using Convolutional Neural Networks (CNNs), that have proven good results in a variety of computer vision tasks, including medical imaging. The network that segments the lesions consists of a cascaded architecture, which first focuses on the region of the liver in order to segment the lesions on it. Moreover, we train a detector to localize the lesions, and mask the results of the segmentation network with the positive detections. The segmentation architecture is based on DRIU~\cite{maninis2016deep}, a Fully Convolutional Network (FCN) with side outputs that work on feature maps of different resolutions, to finally  benefit from the multi-scale information learned by different stages of the network. The main contribution of this work is the use of a detector to localize the lesions, which we show to be beneficial to remove false positives triggered by the segmentation network. 

![Architecture of the network](https://github.com/imatge-upc/detection-2016-nipsws/blob/master/img/hierarchy.png?raw=true)


![Qualitative results](https://github.com/imatge-upc/detection-2016-nipsws/blob/master/img/HR_sequences.png?raw=true)


[logo-nips]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/logos/nips500x95.png?raw=true "NIPS 2016 logo"

| ![Míriam Bellver][bellver-photo]  | ![Xavier Giro-i-Nieto][giro-photo]  | ![Ferran Marqués][marques-photo]  | ![Jordi Torres][torres-photo]  |
|:-:|:-:|:-:|:-:|
| [Míriam Bellver][bellver-web]  | [Xavier Giro-i-Nieto][giro-web]  |  [Ferran Marques][marques-web] | [Jordi Torres][torres-web]  |


[bellver-web]: https://www.bsc.es/bellver-bueno-miriam
[giro-web]: https://imatge.upc.edu/web/people/xavier-giro
[torres-web]: http://www.jorditorres.org/
[marques-web]:https://imatge.upc.edu/web/people/ferran-marques

[bellver-photo]:  https://github.com/imatge-upc/detection-2016-nipsws/blob/master/authors/MiriamBellver160x160.jpg?raw=true "Míriam Bellver"
[giro-photo]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/authors/XavierGiro160x160.jpg?raw=true "Xavier Giró-i-Nieto"
[marques-photo]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/authors/FerranMarques160x160.jpg?raw=true "Ferran Marqués"
[torres-photo]:  https://github.com/imatge-upc/detection-2016-nipsws/blob/master/authors/JordiTorres.jpg?raw=true  "Jordi Torres"

A joint collaboration between:

|![logo-bsc] | ![logo-gpi]  |
|:-:|:-:|
| [Barcelona Supercomputing Center][bsc-web] | [UPC Image Processing Group][gpi-web] |

[gpi-web]: https://imatge.upc.edu/web/ 
[bsc-web]: http://www.bsc.es 

[logo-bsc]:https://github.com/imatge-upc/detection-2016-nipsws/blob/master/logos/bsc320x86.jpg?raw=true "Barcelona Supercomputing Center"
[logo-gpi]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/logos/gpi320x70.png?raw=true "UPC Image Processing Group"
[logo-severo]: https://github.com/imatge-upc/detection-2016-nipsws/blob/master/logos/severo_ochoa.png?raw=true "Severo Ochoa"



## Publication

Our workshop paper is available on [arXiv](https://arxiv.org/abs/1611.03718), and related slides [here](http://www.slideshare.net/xavigiro/hierarchical-object-detection-with-deep-reinforcement-learning).

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

## Code Instructions


### Setup


[VGG-16 weights]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/vgg16_weights.h5
[Image Zooms model]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/model_image_zooms_2


### Usage

##### Training



##### Testing

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


## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/detection-2016-nipsws/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:miriam.bellver@bsc.es>.
