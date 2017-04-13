require 'nn'
require './MyAdd'

function PosELU()
	return nn.Sequential()
		:add(nn.ELU(1,true))
		:add(nn.MyAdd(-1))
end

return PosELU
