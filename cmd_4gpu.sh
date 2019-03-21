export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4
export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v13/

### for images/gpu = 1
### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export OUT_CHANNELS=1024
# export NONLOCAL_OUT_CHANNELS=1024
# export INTER_CHANNELS=512

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
# export OUT_CHANNELS=1024
# export NONLOCAL_OUT_CHANNELS=1024
# export INTER_CHANNELS=512

### Resnet 101, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=1024

### Resnet 50, FPN
export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_bs8.yaml
export PRETRAIN_MODEL=../../R-50.pkl
export OUT_CHANNELS=256
export NONLOCAL_OUT_CHANNELS=1024
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
--nonlocal-cls-num-group 2 \
--nonlocal-cls-num-stack 0 \
--nonlocal-reg-num-group 2 \
--nonlocal-reg-num-stack 0 \
--nonlocal-shared-num-group 4 \
--nonlocal-shared-num-stack 2 \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax False \
--nonlocal-use-ffconv True \
--nonlocal-inter-channels $INTER_CHANNELS \
--nonlocal-out-channels $NONLOCAL_OUT_CHANNELS \
--conv-bbox-expand  1.2 \
--fc-bbox-expand  1.0 \
--backbone-out-channels $OUT_CHANNELS \
--maplevel-fc 0 112 224 448 100000 \
--mask-fc 1 1 1 1 \
--maplevel-conv 0 112 224 448 100000 \
--mask-conv 1 1 1 1 \
--mask-loss 0.5 0.5 0.5 0.5 \
--conv-fc-threshold 224



