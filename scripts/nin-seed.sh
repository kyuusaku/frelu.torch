export CUDA_VISIBLE_DEVICES=$3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

for (( i=$4; i<5; i++ ))  
do  
LOG=./log/$1/$2-0.1-rmax-seed$i.log
th main.lua -dataset $1 -nGPU 1 -batchSize 128 -netType $2 -nEpochs 200 -LR 0.1 -manualSeed $i \
2>&1 | tee -i $LOG
done


