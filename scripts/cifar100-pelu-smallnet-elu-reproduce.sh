export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/pelu-smallnet-elu-reproduce-seed4.log
th main.lua -dataset cifar100 -nGPU 1 -batchSize 128 -netType pelu-smallnet-elu -nEpochs 200 -LR 0.01 -manualSeed 4 \
2>&1 | tee -i $LOG
