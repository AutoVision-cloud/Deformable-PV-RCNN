#!/bin/bash

# copy files
cp -r src/deformable_pfe/* OpenPCDet/pcdet/models/backbones_3d/pfe/
cp -r src/pointnet_modules/* OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/
cp -r configs/def_pv_rcnn.yaml OpenPCDet/tools/cfgs/kitti_models/

