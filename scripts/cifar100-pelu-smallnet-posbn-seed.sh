export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

for (( i=0; i<5; i++ ))  
do  
LOG=./log/cifar100/pelu-smallnet-posbn-seed$i.log
th main.lua -dataset cifar100 -nGPU 1 -batchSize 128 -netType pelu-smallnet-posbn -nEpochs 200 -LR 0.01 -manualSeed $i \
2>&1 | tee -i $LOG
done