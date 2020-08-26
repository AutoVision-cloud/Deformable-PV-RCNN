# Deformable PV-RCNN: Improving 3D Object Detection with Learned Deformations

By [Prarthana Bhattacharyya](https://scholar.google.com/citations?user=v6pGkNQAAAAJ&hl=en) and [Krzysztof Czarnecki](https://scholar.google.com/citations?hl=en&user=ZzCpumQAAAAJ).

We provide code support and configuration files to reproduce the results in the paper for
["Deformable PV-RCNN: Improving 3D Object Detection with Learned Deformations"](https://arxiv.org/abs/2008.08766) on KITTI 3D object detection. Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is a clean open-sourced project for benchmarking 3D object detection methods. 


## Usage
a. Clone the repo:
```
git clone --recursive https://github.com/AutoVision-cloud/DeformablePVRCNN
```
b. Copy Deformable PV-RCNN src into OpenPCDet: 
```
sh ./init.sh
```

c. Install OpenPCDet and prepare KITTI data:

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

d. Run experiments with a specific configuration file:

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more about how to train and run inference on this detector.


## Results and models

The results on KITTI 3D Object Detection val are shown in the table below.
* Our models are trained with 8 GTX 1080Ti GPUs and are available for download.


|                                             | Car | Pedestrian | Cyclist  | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PV-RCNN](OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml) | 83.69 | 54.84 | 69.47 | [model-PV-RCNN](https://drive.google.com/file/d/1CXK7LVGU9jPRcygrDReQWhpwax9BJ9hb/view?usp=sharing) |
| [Deformable PV-RCNN](config/def_pv_rcnn.yaml) | 83.30 | 58.33 | 73.46 | [model-def-PV-RCNN](https://drive.google.com/file/d/18YpEEViDFjKdxhTFxo7mdGWdopCMZ28j/view?usp=sharing) |


## Citing Deformable PV-RCNN

```
@misc{bhattacharyya2020deformable,
    title={Deformable PV-RCNN: Improving 3D Object Detection with Learned Deformations},
    author={Prarthana Bhattacharyya and Krzysztof Czarnecki},
    year={2020},
    eprint={2008.08766},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
