# dataset method&network randomseed #GPUs GPU_ID init_learning_rate
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu 0 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu 1 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu 2 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu 3 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu 4 1 $1 0.01

bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu-bn 0 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu-bn 1 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu-bn 2 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu-bn 3 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-prelu-bn 4 1 $1 0.1
