require 'nn'
require './MyAdd'

function PosSReLU(n, numInputDims)
	return nn.Sequential()
		:add(nn.ReLU(true))
		:add(nn.MyAdd(n, -1, false, numInputDims)) -- must be false when using with ReLU inplace
end

return PosSReLU
