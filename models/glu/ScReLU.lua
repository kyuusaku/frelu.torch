require 'nn'
require './MyMul'
require './MyAdd'

function ScReLU()
	return nn.Sequential()
		:add(nn.MyMul(1))
        :add(nn.MyAdd(0))
		:add(nn.ReLU(true))
end

return ScReLU
