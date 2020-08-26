## Getting Started
 


### Test and evaluate the pretrained models
* Go to tools:
```
cd OpenPCDet/tools
```

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* For example:
```shell script
python test.py --cfg_file cfgs/kitti_models/def_pv_rcnn.yaml --batch_size 4 --ckpt ../ckpt/def_pv_rcnn.pth
```



### Train a model
* Train with multiple GPUs:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}  --epochs 80
```
We use `--batch_size` as 16.

* For example:
```shell script
sh scripts/dist_train.sh 8 --cfg_file cfgs/kitti_models/def_pv_rcnn.yaml  --batch_size 16  --epochs 80
```



* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs 50
```
