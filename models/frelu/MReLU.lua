require 'nn'
require './MyMul'

function MReLU(val)
	return nn.Sequential()
		:add(nn.MyMul(val))
		:add(nn.ReLU(true))
end

return MReLU
