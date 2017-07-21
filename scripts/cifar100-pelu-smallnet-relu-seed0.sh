export CUDA_VISIBLE_DEVICES=4,5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/pelu-smallnet-relu-seed0.log
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -netType pelu-smallnet-relu -nEpochs 200 -LR 0.01 -manualSeed 0 \
2>&1 | tee -i $LOG
