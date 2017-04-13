require 'nn'
require 'cudnn'
require 'NewReshape'

model=torch.load('/home/lab-cai.bolun/Project/fb.resnet.torch/checkpoints/model_200.t7')
weights=torch.zeros(110)
for k,v in pairs(model:findModules('nn.Add')) do
	weights[k]=v.bias:double()
	--print(v.weight)
end
print(weights:reshape(55,2))

--[[
data = torch.randn(1,2,8,8)
grad = torch.randn(1,2,8,8)

mlp1 = nn.Sequential()
mlp1:add(nn.Reshape(1,1,2*8,8,false))
print(mlp1:backward(data, grad))

mlp2 = nn.Sequential()
mlp2:add(nn.NewReshape(1,1,2*8,8))
print(mlp2:backward(data, grad))
--]]
