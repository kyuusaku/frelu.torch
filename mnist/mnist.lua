require 'torch'  
require 'nn'  
require 'optim'  
require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

-- Reference: https://github.com/ydwen/caffe-face/tree/caffe-face/mnist_example

print('Read data set')
mnist = require 'mnist' 
trainset = mnist.traindataset()
testset = mnist.testdataset()
trainset = {
    size = 60000,
    data = trainset.data:type('torch.FloatTensor'),
    label = trainset.label
}
testset = {
    size = 10000,
    data = testset.data:type('torch.FloatTensor'),
    label = testset.label
}
print('Normalize')
mean = 127.5
scale = 0.0078125
trainset.data[{ {}, {}, {} }]:add(-mean)
trainset.data[{ {}, {}, {} }]:mul(scale)
testset.data[{ {}, {}, {}  }]:add(-mean)
testset.data[{ {}, {}, {}  }]:mul(scale)
print(trainset)
print(testset)
print(testset.data[{1,{},{}}])

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU
convblock = function(ninput, noutput)
   return nn.Sequential()
      :add(Convolution(ninput,noutput,5,5,1,1,2,2))
      :add(ReLU(true))
      :add(Convolution(noutput,noutput,5,5,1,1,2,2))
      :add(ReLU(true))
      :add(Max(2,2,2,2,0,0))
end

create_model = function()
    local net = nn.Sequential()
    net:add(nn.Reshape(1,28,28))
    net:add(convblock(1,32))
    net:add(convblock(32,64))
    net:add(convblock(64,128))
    net:add(nn.View(3*3*128))
    net:add(nn.Linear(3*3*128,2))
    net:add(ReLU(true))
    net:add(nn.Linear(2,10))

    local function ConvInit(name)   
      for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(1/n))
         v.bias:zero()
      end
    end

    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
    for k,v in pairs(net:findModules('nn.Linear')) do
      v.bias:zero()
    end
    net:cuda()

    return net
end

print('Create model')
model = create_model()
print(model)
criterion = nn.CrossEntropyCriterion():cuda()

print('Start training')
learningRate = function(epoch)
    local base_lr = 0.01
    local gamma = 0.8
    local decay = 0
    decay = epoch >= 62 and 2 or epoch >= 40 and 1 or 0
    return base_lr * math.pow(gamma, decay)
end

sgd_params = {
    learningRate = 0.01,
    learningRateDecay = 0.0,
    weightDecay = 0.0005,
    momentum = 0.9
}

--[[ flatten parameters
optim expects the parameters that are to be optimized, and
their gradients, to be one-dimensional Tensors.
--]]
x, dl_dx = model:getParameters()

step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 128
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size, trainset.size) - t
        local inputs = torch.CudaTensor(size, 28, 28)
        local targets = torch.CudaTensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t-1]]
            local target = trainset.label[shuffle[i+t-1]]
            inputs[i] = input:cuda()
            targets[i] = target
        end
        targets:add(1)
        -- 
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end 
            dl_dx:zero()
            -- perform mini-batch gradient descent
            model:forward(inputs)
            local loss = criterion:forward(model.output, targets)
            criterion:backward(model.output, targets)
            model:backward(inputs, criterion.gradInput)
            -- return loss & accumulation of the gradients
            return loss, dl_dx
        end
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end
    --normalize loss
    return current_loss / count
end

eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 100
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]:cuda()
        local targets = torch.CudaLongTensor()
        label = dataset.label[{{i,i+size-1}}]
        targets:resize(label:size()):copy(label)
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        indices:double()
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end
    return count / dataset.size
end

--[[train the model.
after each epoch, evaluate the accuracy on the validation dataset,
in order to decide whether to stop
--]]
max_iters = 78
do
    for i = 1,max_iters do
        sgd_params.learningRate = learningRate(i)
        local loss = step()
        print(string.format('Epoch: %d, learningRate: %4f, Current loss: %4f', i, sgd_params.learningRate, loss))
        local accuracy = eval(testset)
        print(string.format('Accuracy on the test set: %4f', accuracy))
    end
end
