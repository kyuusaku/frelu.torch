export CUDA_VISIBLE_DEVICES=6,7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/pelu-smallnet-relu-bn-seed1.log
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -netType pelu-smallnet-relu-bn -nEpochs 200 -LR 0.01 -manualSeed 1 \
2>&1 | tee -i $LOG
