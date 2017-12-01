require 'nn'
require './MyAdd'

function PreSReLU()
	return nn.Sequential()
		:add(nn.MyAdd(1))
		:add(nn.ReLU(true))
end

return PreSReLU
