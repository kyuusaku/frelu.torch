export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar10/resnet-srelu-20-seed0.log
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -netType resnet-srelu -nEpochs 200 -depth 20 -manualSeed 0 \
2>&1 | tee -i $LOG
