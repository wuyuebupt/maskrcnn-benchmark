export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_4gpu_neighbor.yaml --output-dir --output-dir /work/maskrcnn/iccv19/model_output_tmp/  --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/ --nonlocal-num-group 2 --nonlocal-num-stack 1 --bbox-expand  1.2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_4gpu.yaml --output-dir /work/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/ --nonlocal-num-group 2 --nonlocal-num-stack 1 --bbox-expand  1.2

