export CUDA_VISIBLE_DEVICES=4,5,6,7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/cifar10/resnet-56-nocudnnall-run2.log
th main.lua -dataset cifar10 -nGPU 4 -batchSize 128 -netType resnet -nEpochs 200 -depth 56 -nThreads 8 \
2>&1 | tee -i $LOG
