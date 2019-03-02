export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4

### for images/gpu = 1
export CONFIG_YAML=configs/e2e_faster_rcnn_R_50_C4_1x_4gpu_neighbor_bs4.yaml

### for images/gpu = 2
# export CONFIG_YAML=configs/e2e_faster_rcnn_R_50_C4_1x_4gpu_neighbor_bs8.yaml

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--output-dir /work/maskrcnn/iccv19/model_output_tmp_v3/  \
--pretrained-model ../../R-50.pkl \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-cls-num-group 1 \
--nonlocal-cls-num-stack 0 \
--nonlocal-reg-num-group 1 \
--nonlocal-reg-num-stack 0 \
--nonlocal-use-bn True \
--bbox-expand  1.2



# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_4gpu.yaml --output-dir /work/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/ --nonlocal-num-group 2 --nonlocal-num-stack 1 --bbox-expand  1.2

