require 'nn'
require './MyAdd'

function PosSReLU()
	return nn.Sequential()
		:add(nn.ReLU(true))
		:add(nn.MyAdd(-1))
end

return PosSReLU
