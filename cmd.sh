export PYTHONPATH=$PWD/maskrcnn_pythonpath

export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v6/

### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export INTER_CHANNELS=1024

### Resnet 101, C4
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=1024

### Resnet 50, FPN
export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor.yaml
export PRETRAIN_MODEL=../../R-50.pkl
export INTER_CHANNELS=256

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=256


python  tools/train_net.py \
--output-dir $OUTPUT_DIR  \
--pretrained-model $PRETRAIN_MODEL \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-cls-num-group 1 \
--nonlocal-cls-num-stack 1 \
--nonlocal-reg-num-group 1 \
--nonlocal-reg-num-stack 1 \
--nonlocal-shared-num-group 1 \
--nonlocal-shared-num-stack 1 \
--nonlocal-inter-channels $INTER_CHANNELS \
--nonlocal-use-shared True \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax True \
--bbox-expand  1.2
