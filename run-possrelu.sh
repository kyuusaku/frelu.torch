# dataset method&network randomseed #GPUs GPU_ID init_learning_rate
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu 0 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu 1 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu 2 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu 3 1 $1 0.01
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu 4 1 $1 0.01

bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu-bn 0 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu-bn 1 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu-bn 2 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu-bn 3 1 $1 0.1
bash scripts/plain-seed.sh cifar100 pelu-smallnet-possrelu-bn 4 1 $1 0.1