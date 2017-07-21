export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/pelu-smallnet-psrelu-seed0.log
th main.lua -dataset cifar100 -nGPU 4 -batchSize 128 -netType pelu-smallnet-psrelu -nEpochs 200 -LR 0.01 -manualSeed 0 \
2>&1 | tee -i $LOG
