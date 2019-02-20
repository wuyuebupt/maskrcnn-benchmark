export PYTHONPATH=$PWD/maskrcnn_pythonpath

export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x_4gpu.yaml
