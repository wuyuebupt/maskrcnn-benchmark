export PYTHONPATH=$PWD/maskrcnn_pythonpath

# python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_neighbor.yaml  --output-dir /work/maskrcnn/iccv19/model_output_tmp_v2/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/ --nonlocal-num-group 2 --nonlocal-num-stack 3 --bbox-expand  1.2


export CONFIG_YAML=configs/e2e_faster_rcnn_R_50_C4_1x_neighbor.yaml

python  tools/train_net.py \
--output-dir /work/maskrcnn/iccv19/model_output_tmp_v3/  \
--pretrained-model ../../R-50.pkl \
--data-dir ../maskrcnn-benchmark-file/datasets/ \
--config-file $CONFIG_YAML \
--nonlocal-cls-num-group 1 \
--nonlocal-cls-num-stack 1 \
--nonlocal-reg-num-group 1 \
--nonlocal-reg-num-stack 1 \
--nonlocal-use-bn False \
--nonlocal-use-relu True \
--bbox-expand  1.2

# python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x.yaml --output-dir /work/maskrcnn/iccv19/model_output_tmp/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/
# python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C5_1x.yaml --output-dir /work/maskrcnn/iccv19/model_output_tmp/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/
