export CUDA_VISIBLE_DEVICES=4,5,6,7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar10/resnet-glu-20-0.0002-A-run1.log
th main.lua -dataset cifar10 -nGPU 4 -batchSize 128 -netType resnet-glu -nEpochs 200 -depth 20 -weightDecay 0.0002 -nThreads 8 -cudnn deterministic -shortcutType A \
2>&1 | tee -i $LOG
