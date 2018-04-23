# dataset method&network randomseed #GPUs GPU_ID init_learning_rate
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu 0 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu 1 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu 2 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu 3 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu 4 1 $1 0.01

bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu-bn 0 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu-bn 1 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu-bn 2 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu-bn 3 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-relu-bn 4 1 $1 0.1