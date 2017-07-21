export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/pelu-smallnet-selu-init1-seed0-0.1.log
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -netType pelu-smallnet-selu -nEpochs 200 -LR 0.1 -manualSeed 0 \
2>&1 | tee -i $LOG
