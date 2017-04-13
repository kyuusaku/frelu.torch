require 'nn'
require './NewReshape'
require './ConstrainedMul'

function WELU(count, dimension)
	local function Scale()
		local s = nn.Sequential()
		s:add(nn.Mul())
		--s:add(nn.ConstrainedMul(true))
		s:add(nn.Add(1,true))
		return s
	end
	local function ScaleConstant(k,b)
		local s = nn.Sequential()
		s:add(nn.MulConstant(k))
		s:add(nn.AddConstant(b))
		return s
	end
	local function MulLinear(count)
		local d = nn.Concat(2)
		--
		for i=1,count do
			d:add(Scale())
		end
		--]]
		--[[
		d:add(ScaleConstant(0.97,0.23))
		d:add(Scale())
		--]]
		return d
	end
	
	return nn.Sequential()
		:add(nn.NewReshape(1,-1,0))
		:add(MulLinear(count))
		:add(nn.ConcatTable()
			:add(nn.Identity())
			:add(nn.SpatialSoftMax()))
        :add(nn.CMulTable())
		:add(nn.Sum(2))
		:add(nn.Unsqueeze(2))
		:add(nn.NewReshape(dimension,-1,0))
end

return WELU
