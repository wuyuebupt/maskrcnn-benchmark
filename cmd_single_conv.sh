export PYTHONPATH=$PWD/maskrcnn_pythonpath

# export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v15/
# export OUTPUT_DIR=/work/maskrcnn/iccv19/model_output_tmp_v15/
### resnet 101
# export OUTPUT_DIR=/work/dataforYinpeng/saw_models_toeval/nips/fpn101-1x-DHNew-bc13-back256-conv-att-Ys200g4-c1024x512-L112x224x448-m125x100-wc04x16-f14x06--wu1-input-1553839402050_9330
### resnet 50
export OUTPUT_DIR=/work/dataforYinpeng/saw_models_toeval/single_head_eval/fpn50-1x-dh-b12-back256-conv-only-Ys400g4-c1024x512-112x224x448-m10-wc10x10-f00x00-sc2-input-1553303475036_3836/



### Resnet 50, C4
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_50_C4_1x_neighbor.yaml
# export PRETRAIN_MODEL=../../R-50.pkl
# export NONLOCAL_OUT_CHANNELS=2048
# export INTER_CHANNELS=1024
# export OUT_CHANNELS=1024

### Resnet 101, C4
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_101_C4_1x_neighbor.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export INTER_CHANNELS=1024

### Resnet 50, FPN
export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor.yaml
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_50_FPN_1x_neighbor_testdev.yaml

export PRETRAIN_MODEL=../../R-50.pkl
export OUT_CHANNELS=256
export NONLOCAL_OUT_CHANNELS=1024
export INTER_CHANNELS=512

### Resnet 101, FPN
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor.yaml
# export CONFIG_YAML=configs/bbox_expand_1gpu/e2e_faster_rcnn_R_101_FPN_1x_neighbor_testdev.yaml
# export PRETRAIN_MODEL=../../R-101.pkl
# export OUT_CHANNELS=256
# export NONLOCAL_OUT_CHANNELS=1024
# export INTER_CHANNELS=512

# export OUT_CHANNELS=2048
# export INTER_CHANNELS=1024


python  tools/train_net.py \
--output-dir $OUTPUT_DIR  \
--pretrained-model $PRETRAIN_MODEL \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-cls-num-group 4 \
--nonlocal-cls-num-stack 0 \
--nonlocal-reg-num-group 4 \
--nonlocal-reg-num-stack 0 \
--nonlocal-shared-num-group 4 \
--nonlocal-shared-num-stack 4 \
--nonlocal-inter-channels $INTER_CHANNELS \
--nonlocal-out-channels $NONLOCAL_OUT_CHANNELS \
--nonlocal-use-bn True \
--nonlocal-use-relu True \
--nonlocal-use-softmax False \
--nonlocal-use-ffconv True \
--nonlocal-use-attention False \
--conv-bbox-expand  1.2 \
--fc-bbox-expand  1.0 \
--backbone-out-channels $OUT_CHANNELS \
--maplevel-fc 0 112 224 448 100000 \
--mask-fc 1 1 1 1 \
--maplevel-conv 0 112 224 448 100000 \
--mask-conv 1 1 1 1 \
--mask-loss 0.5 0.5 0.5 0.5 \
--conv-fc-threshold 224 \
--lr-steps 100 200 300 \
--stop-gradient 1 1 1 1 \
--evaluation-flags 1 0 0 0

###### --stop-gradient
# conv cls
# conv reg
# fc   cls
# fc   reg

###### --evaluation-flags: 4 evaluations, in order, 0 off, 1 on 
# 0 : conv cls + conv reg
# 0 : fc cls + fc cls
# 0 : fc cls + conv reg
# 0 : fc cls + conv reg (in posterior bayesian manner)
# conv reg
# fc   cls

# --lr-steps 120000 160000 180000


####### lr schedule
# 1x, 180k: decrease at 120000 and 160000, end at 180000
#
#
#
# 

####### for mask
#         self.conv_cls_weight = mask_loss[0]
#         self.conv_reg_weight = mask_loss[1]
#         self.fc_cls_weight = mask_loss[2]
#         self.fc_reg_weight = mask_loss[3]


# --maplevel-fc 0 112 224 448 100000 \
# --maplevel-conv 0 0 160 320 100000
