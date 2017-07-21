require 'nn'
require 'cunn'
require './MyMul'
require './MyAdd'
require './BS'

mlp1 = nn.Sequential()
mlp1:add(nn.ReLU(true))
mlp1:add(nn.MyAdd(1, -1, false))
mlp1:add(nn.MyMul(1, 1, true))
mlp1:cuda()

mlp2 = nn.Sequential()
mlp2:add(nn.ReLU(true))
mlp2:add(nn.MyAdd(1, -1, false))
mlp2:add(nn.MyMul(1, 1, false))
mlp2:cuda()

mlp3 = nn.Sequential()
mlp3:add(nn.ReLU(true))
mlp3:add(nn.BS(-1, 1))
mlp3:cuda()

x = torch.rand(5)
y = x:clone()
y:add(math.pi)

pred1 = mlp1:forward(x)
pred2 = mlp2:forward(x)
pred3 = mlp3:forward(x)
print(pred1)
print(pred2)
print(pred3)

gradC1 = nn.MSECriterion():backward(pred1, y)
gradC2 = nn.MSECriterion():backward(pred2, y)
gradC3 = nn.MSECriterion():backward(pred3, y)
print(gradC1)
print(gradC2)
print(gradC3)

mlp1:zeroGradParameters()
mlp2:zeroGradParameters()
mlp3:zeroGradParameters()
mlp1:backward(x, gradC1)
mlp2:backward(x, gradC2)
mlp3:backward(x, gradC3)
print(mlp1:get(3).gradWeight)
print(mlp2:get(3).gradWeight)
print(mlp3:get(2).gradWeight)
print(mlp1:get(2).gradBias)
print(mlp2:get(2).gradBias)
print(mlp3:get(2).gradBias)