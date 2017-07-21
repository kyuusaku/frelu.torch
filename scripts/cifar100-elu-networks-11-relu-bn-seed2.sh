export CUDA_VISIBLE_DEVICES=2,3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/elu-networks-11-relu-bn-seed2.log
th main-elu.lua -dataset cifar100 -nGPU 2 -batchSize 100 -netType elu-networks-relu-bn -nEpochs 330 -weightDecay 0.0005 -manualSeed 2 \
2>&1 | tee -i $LOG
