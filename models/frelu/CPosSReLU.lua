require 'nn'

function CPosSReLU(val)
	return nn.Sequential()
		:add(nn.ReLU(true))
		:add(nn.AddConstant(val))
end

return CPosSReLU
