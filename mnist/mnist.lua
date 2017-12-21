require 'torch'  
require 'nn'  
require 'optim'  

print('Read data set')
mnist = require 'mnist' 
fullset = mnist.traindataset()
testset = mnist.testdataset()

trainset = {
    size = 50000,
    data = fullset.data[{{1,50000}}]:double(),
    label = fullset.label[{{1,50000}}]
}

validationset = {
    size = 10000,
    data = fullset.data[{{50001,60000}}]:double(),
    label = fullset.label[{{50001,60000}}]
}

print('Normalize')
mean = {}
stdv  = {}
mean = trainset.data[{ {}, {}, {}}]:mean()
trainset.data[{ {}, {}, {}  }]:add(-mean)
stdv = trainset.data[{ {}, {}, {}  }]:std()
trainset.data[{ {}, {}, {}  }]:div(stdv)

testset.data[{ {}, {}, {}  }]:add(-mean)
testset.data[{ {}, {}, {}  }]:div(stdv)

create_model = function(ACT)
    local net = nn.Sequential()
    net:add(nn.View(-1,1,28,28))
    net:add(nn.SpatialConvolution(1,32,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialConvolution(32,32,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
    net:add(nn.SpatialConvolution(32,64,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialConvolution(64,64,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
    net:add(nn.SpatialConvolution(64,128,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialConvolution(128,128,5,5,1,1,2,2))
    net:add(ACT)
    net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
    net:add(nn.View(-1):setNumInputDims(3))
    net:add(nn.Linear(3*3*128,2))
    net:add(ACT)
    net:add(nn.Linear(2,10))
    return net
end

print('Create model')
model = create_model(nn.ReLU(True))
print(model)
criterion = nn.CrossEntropyCriterion()

model = require('weight-init')(model, 'xavier') 

--[[ use optim package to train the network.
optim contains several optimization algorithms.
all of these algorithms assume the same parameters:
* a closure that computes the loss, and its gradient wrt to x, given a point x
* a point x
* some parameters, which are algorithm-specific
--]]

sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 1e-3,
    momentum = 1e-4
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
    batch_size = batch_size or 200
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        local targets = torch.Tensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t-1]]
            local target = trainset.label[shuffle[i+t-1]]
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        -- 
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end 
            dl_dx:zero()
            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))
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
    batch_size = batch_size or 200
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end
    return count / dataset.size
end

--[[train the model.
after each epoch, evaluate the accuracy on the validation dataset,
in order to decide whether to stop
--]]
max_iters = 30
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

testset.data = testset.data:double()
print(string.format('Accuracy on the test set: %4f', eval(testset)))