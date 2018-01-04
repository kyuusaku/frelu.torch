export CUDA_VISIBLE_DEVICES=$4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/imagenet/$1-$2-2.log
th main.lua -netType $1 -depth $2 -nGPU $3 -nThreads $5 -batchSize 256 -data /home/lab/Dataset/ImageNet/ -save checkpoints-imagenet-$1-2 \
2>&1 | tee -i $LOG
