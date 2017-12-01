require 'nn'
require './MyAdd'

function DSReLU()
	return nn.Sequential()
		:add(nn.MyAdd(1,1,false))
		:add(nn.ReLU(true))
		:add(nn.MyAdd(1,-1,false))
end

return DSReLU
