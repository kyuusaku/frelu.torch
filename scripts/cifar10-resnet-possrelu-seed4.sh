export CUDA_VISIBLE_DEVICES=4,5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar10/resnet-possrelu-$1-seed4.log
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -netType resnet-possrelu -nEpochs 200 -depth $1 -manualSeed 4 \
2>&1 | tee -i $LOG
