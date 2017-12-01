require 'nn'
require 'MyMul'

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.MyMul(1, 1, false))
for i = 1, 10000 do
   x = torch.rand(5)
   y:copy(x)
   y:mul(math.pi)
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end
print('MyMul(1, 1, false)')
print(mlp:get(1).weight)

y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.MyMul(1, 1, true))
for i = 1, 10000 do
   x = torch.rand(5)
   y:copy(x)
   y:mul(math.pi)
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end
print('MyMul(1, 1, true)')
print(mlp:get(1).weight)

y = torch.Tensor(2,2)
mlp = nn.Sequential()
mlp:add(nn.MyMul(2, 1, false, 2))
for i = 1, 10000 do
   x = torch.rand(2,2)
   y:copy(x)
   for j = 1, 2 do
      y[j]:mul(j)
   end
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end
print('MyMul(2, 1, false, 2)')
print(mlp:get(1).weight)

y = torch.Tensor(2,2)
mlp = nn.Sequential()
mlp:add(nn.MyMul(2, 1, false, 1))
for i = 1, 10000 do
   x = torch.rand(2,2)
   y:copy(x)
   for j = 1, 2 do
      y[{{},j}]:mul(j)
   end
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end
print('MyMul(2, 1, false, 1)')
print(mlp:get(1).weight)

y = torch.Tensor(2,2)
mlp = nn.Sequential()
mlp:add(nn.MyMul(2, 1, false))
for i = 1, 10000 do
   x = torch.rand(2,2)
   y:copy(x)
   for j = 1, 2 do
      y[{{},j}]:mul(j)
   end
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end
print('MyMul(2, 1, false)')
print(mlp:get(1).weight)
