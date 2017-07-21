export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/elu-networks-11-elu-seed4.log
th main-elu.lua -dataset cifar100 -nGPU 2 -batchSize 100 -netType elu-networks-elu -nEpochs 330 -weightDecay 0.0005 -manualSeed 4 \
2>&1 | tee -i $LOG
