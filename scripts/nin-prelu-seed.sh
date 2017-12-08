export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

for (( i=0; i<5; i++ ))  
do  
LOG=./log/cifar100/nin-prelu-bn-0.1-seed$i.log
th main.lua -dataset cifar100 -nGPU 1 -batchSize 128 -netType nin-prelu-bn -nEpochs 200 -LR 0.1 -manualSeed $i \
2>&1 | tee -i $LOG
done


