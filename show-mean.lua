require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
PosSReLU = require './models/frelu/PosSReLU'
local DataLoader = require 'dataloader'

local function parse(arg)
	local cmd = torch.CmdLine()
	cmd:option('-model',      'none',   'Path to model to show')
    cmd:option('-dataset',    'cifar100', 'Test dataset')
	local opt = cmd:parse(arg or {})
    opt.manualSeed = 0
    opt.gen = 'gen'
    opt.nThreads = 2
    opt.testOnly = true
    opt.tenCrop = false
    opt.batchSize = 100
	return opt
end

local function copyInputs(sample)
   -- Copies the input to a CUDA tensor
   input = torch.CudaTensor()
   target = torch.CudaTensor()
   input:resize(sample.input:size()):copy(sample.input)
   target:resize(sample.target:size()):copy(sample.target)
   return input, target
end

local function test(model, dataloader)
    sums = torch.FloatTensor(4):fill(0)
    squared_sums = torch.FloatTensor(4):fill(0)
    model:evaluate()
    local size = dataloader:size()
    local N = 0
    for n, sample in dataloader:run() do   
        -- Copy input and target to the GPU
        input, target = copyInputs(sample)
        local batchSize = sample:size(1)
        N = N + batchSize

        model:forward(input)

        layer1 = model:get(1):get(2).output:float()
        layer2 = model:get(2):get(2).output:float()
        layer3 = model:get(3):get(2).output:float()
        layer4 = model:get(6).output:float()

        sums[1] = sums[1] + layer1:sum()
        sums[2] = sums[2] + layer2:sum()
        sums[3] = sums[3] + layer3:sum()
        sums[4] = sums[4] + layer4:sum()
        squared_sums[1] = squared_sums[1] + torch.power(layer1,2):sum()
        squared_sums[2] = squared_sums[2] + torch.power(layer2,2):sum()
        squared_sums[3] = squared_sums[3] + torch.power(layer3,2):sum()
        squared_sums[4] = squared_sums[4] + torch.power(layer4,2):sum()

        print((' | Test: [%d/%d]'):format(n, size))

    end
    means = sums / N
    stds = torch.sqrt(squared_sums / N - torch.pow(means, 2))
    return means, stds
end

local opt = parse(arg)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

local model = torch.load(opt.model):cuda()
-- First remove any DataParallelTable
if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
end
print(model)

means, stds = test(model, valLoader)
print(means)
print(stds)