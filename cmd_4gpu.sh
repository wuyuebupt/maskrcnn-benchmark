export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4
# export OUTPUT_DIR=/work/dataforYinpeng/saw_models_toeval/double-head/fpn50-1x-dh-b12-back256-conv-only-Ys300g4-c1024x512-112x224x448-m10-wc10x10-f00x00-sc2-input-1553303475036_3797/
export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v19/

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
--nonlocal-cls-num-group 4 \
--nonlocal-cls-num-stack 0 \
--nonlocal-reg-num-group 4 \
--nonlocal-reg-num-stack 0 \
--nonlocal-shared-num-group 4 \
--nonlocal-shared-num-stack 2 \
--nonlocal-fc-num-stack 2 \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax False \
--nonlocal-use-ffconv True \
--nonlocal-use-attention False \
--fc-use-ffconv False \
--fc-use-attention False \
--nonlocal-inter-channels $INTER_CHANNELS \
--nonlocal-out-channels $NONLOCAL_OUT_CHANNELS \
--conv-bbox-expand  1.2 \
--fc-bbox-expand  1.0 \
--backbone-out-channels $OUT_CHANNELS \
--maplevel-fc 0 112 224 448 100000 \
--mask-fc 1 1 1 1 \
--maplevel-conv 0 112 224 448 100000 \
--mask-conv 1 1 1 1 \
--mask-loss 0.4 1.6 1.6 0.4 \
--stop-gradient 1 1 1 1 \
--evaluation-flags 1 1 1 1 \
--lr-steps 100 200 300
# --lr-steps 120000 160000 180000

####### --stop-gradient 1 0 1 0: 4 flags in order, 0 off no gradient, 1 with gradient ########
# conv cls
# conv reg
# fc   cls
# fc   reg
##############################################################################################

####### --evaluation-flags: 4 evaluations in order, 0 off, 1 on, can use 1 1 1 1 #############
# conv cls + conv reg
# fc cls + fc cls
# fc cls + conv reg
# fc cls + conv reg (in posterior bayesian manner)
##############################################################################################

####### lr schedule: with batch size 8, init lr 0.01 ##################
# 1x, 180k: decrease at 120000 and 160000, end at 180000, This schedules results in 12.17 epochs over the 118,287 images in coco_2014_train union coco_2014_valminusminival (or equivalently, coco_2017_train).
# --lr-steps 120000 160000 180000
# 
# 2x: twice
# --lr-steps 240000 320000 360000
#
# s1x: stretched 1x, 1.44x + extends the duration of the first learning rate
# --lr-steps 200000 240000 260000 
#
# 280k: cascade RCNN, (the setting used: FPN+(RoIAlign) , ResNet101, 512 ROIs, batchsize 8, init lr 0.005)
# --lr-steps 160000 240000 280000
#######################################################################





#### for debug, 
# --lr-steps 100 200 300
# SOLVER:
#   BASE_LR: 0.01
#   IMS_PER_BATCH: 8
#   MAX_ITER: 300
#   STEPS: (100, 200)




####### for mask
#         self.conv_cls_weight = mask_loss[0]
#         self.conv_reg_weight = mask_loss[1]
#         self.fc_cls_weight = mask_loss[2]
#         self.fc_reg_weight = mask_loss[3]
