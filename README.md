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


## Results and models

The results on KITTI 3D Object Detection val are shown in the table below.
