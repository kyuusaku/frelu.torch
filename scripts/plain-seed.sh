export CUDA_VISIBLE_DEVICES=$5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/$1/$2-seed$3.log
th main.lua -dataset $1 -nGPU $4 -batchSize 128 -netType $2 -nEpochs 200 -manualSeed $3 -LR $6 \
2>&1 | tee -i $LOG
