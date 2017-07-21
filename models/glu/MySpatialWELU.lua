require 'nn'
require './NewReshape'
require './MyMul'
require './MyAdd'

function MySpatialWELU(dimension)
   local function MyScale(m,a)
      local s = nn.Sequential()
      s:add(nn.MyMul(m))
      s:add(nn.MyAdd(a))
      return s
   end

   local function MulLinear()
      local d = nn.Concat(2)
      d:add(MyScale(1,0))
      d:add(MyScale(0.01,-1))
      return d
   end

   return nn.Sequential()
      :add(nn.NewReshape(1,-1,0))
      :add(MulLinear())
      :add(nn.ConcatTable()
	 :add(nn.Identity())
	 :add(nn.SpatialSoftMax()))
      :add(nn.CMulTable())
      :add(nn.Sum(2))
      :add(nn.Unsqueeze(2))
      :add(nn.NewReshape(dimension,-1,0))
end

return MySpatialWELU
