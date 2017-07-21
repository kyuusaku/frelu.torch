export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/elu-networks-11-psrelu-seed0.log
th main-elu.lua -dataset cifar100 -nGPU 8 -batchSize 100 -netType elu-networks-psrelu -nEpochs 330 -weightDecay 0.0005 -manualSeed 0 \
2>&1 | tee -i $LOG
