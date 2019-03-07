export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4
export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v6/

### for images/gpu = 1
### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=1024

### Resnet 101, C4
export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor_bs4.yaml
export PRETRAIN_MODEL=../../R-101.pkl
export INTER_CHANNELS=1024

### Resnet 50, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=256

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor_bs4.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=256



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
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=256

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_4gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor_bs8.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=256


python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--output-dir $OUTPUT_DIR  \
--pretrained-model $PRETRAIN_MODEL \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-cls-num-group 2 \
--nonlocal-cls-num-stack 3 \
--nonlocal-reg-num-group 2 \
--nonlocal-reg-num-stack 3 \
--nonlocal-shared-num-group 1 \
--nonlocal-shared-num-stack 1 \
--nonlocal-use-shared True \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax True \
--nonlocal-inter-channels $INTER_CHANNELS \
--bbox-expand  1.2



# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_4gpu.yaml --output-dir /work/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/ --nonlocal-num-group 2 --nonlocal-num-stack 1 --bbox-expand  1.2

