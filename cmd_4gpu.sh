export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4
export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v8/

### for images/gpu = 1
### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=1024

### Resnet 101, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=1024

### Resnet 50, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export OUT_CHANNELS=2048
# export INTER_CHANNELS=1024

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export OUT_CHANNELS=2048
# export INTER_CHANNELS=1024



### for images/gpu = 2
### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=1024

### Resnet 101, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=1024

### Resnet 50, FPN
export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_bs8.yaml
export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=1024
### out channels is now for fpn out channels, not the original setging, no need to change the yaml config file 
export OUT_CHANNELS=512
### inter channels is still used in each non local module
export INTER_CHANNELS=512

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export OUT_CHANNELS=2048
# export INTER_CHANNELS=1024


python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--output-dir $OUTPUT_DIR  \
--pretrained-model $PRETRAIN_MODEL \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-shared-num-group 1 \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax False \
--nonlocal-use-ffconv True \
--nonlocal-inter-channels $INTER_CHANNELS \
--bbox-expand  1.0 \
--fpn-out-channels $OUT_CHANNELS \
--mode-code 0 3 0
## for no non-local at all, comment the model-code or leave an empty input

### mode-code codebook
### input  : x (cls), y (reg)
### output : x (cls), y (reg)
### code | attention input and output
### ----------------------------------
### 0    | x <- (x,x) y <-(y,y)
### 1    | x <- (x,y) y <-(y,y)
### 2    | x <- (x,x) y <-(y,x)
### 3    | x <- (x,y) y <-(y,x)
### ----------------------------------
### demos
### no nonlocal: leave it empty or use default value
### one seperate, followed by a double cross attention: 0 3
### seperate-double cross-seperate: 0 3 0

