export PYTHONPATH=$PWD/maskrcnn_pythonpath

python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_C4_1x.yaml --output-dir /work/ --pretrained-model ../../R-50.pkl --data-dir ../maskrcnn-benchmark-file/datasets/
