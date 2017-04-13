require 'nn'
require './MyAdd'

function DSReLU()
	return nn.Sequential()
		:add(nn.MyAdd(1))
		:add(nn.ReLU(true))
		:add(nn.MyAdd(-1))
end

return DSReLU
