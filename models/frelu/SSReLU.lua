require 'nn'
require './MyConstrainedMul'
require './MyAdd'

function SSReLU(n)
	return nn.Sequential()
--                :add(nn.MyConstrainedMul(1))
		:add(nn.ReLU(true))
		:add(nn.MyAdd(1, -0.5, false)) -- must be false when using with ReLU inplace
--		:add(nn.MyConstrainedMul(1))

end

return SSReLU
