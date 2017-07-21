export CUDA_VISIBLE_DEVICES=$3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar100/resnet-$1-seed$2.log
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -netType resnet -nEpochs 200 -depth $1 -manualSeed $2 \
2>&1 | tee -i $LOG
