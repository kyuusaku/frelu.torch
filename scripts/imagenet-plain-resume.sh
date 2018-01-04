export CUDA_VISIBLE_DEVICES=$3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

set -x
set -e
export PYTHONUNBUFFERED="True"

LOG=./log/imagenet/$1-resume.log
th main.lua -netType $1 -nGPU $2 -nThreads $4 -batchSize 256 -data /home/lab/Dataset/ImageNet/ -save checkpoints-imagenet-$1 -resume checkpoints-imagenet-$1 \
2>&1 | tee -i $LOG
