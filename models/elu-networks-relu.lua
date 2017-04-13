local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU
local Dropout = nn.Dropout

local function createModel(opt)
   local model = nn.Sequential()
   model:add(Convolution(3,192,5,5,1,1,2,2))--1x192x5 32x32
   model:add(ReLU(true))
   --stack
   model:add(Convolution(192,192,1,1))--1x192x1 32x32
   model:add(ReLU(true))
   model:add(Convolution(192,240,3,3,1,1,1,1))--1x240x3
   model:add(ReLU(true))
   model:add(Dropout(0.1))
   model:add(Max(2,2,2,2))
   --stack
   model:add(Convolution(240,240,1,1))--1x240x1 16x16
   model:add(ReLU(true))
   model:add(Convolution(240,260,2,2))--1x260x2 15x15
   model:add(ReLU(true))
   model:add(Dropout(0.2))
   model:add(Max(2,2,2,2))
   --stack
   model:add(Convolution(260,260,1,1))--1x260x1 7x7
   model:add(ReLU(true))
   model:add(Convolution(260,280,2,2))--1x280x2 6x6
   model:add(ReLU(true))
   model:add(Dropout(0.3))
   model:add(Max(2,2,2,2))
   --stack
   model:add(Convolution(280,280,1,1))--1x280x1 3x3
   model:add(ReLU(true))
   model:add(Convolution(280,300,2,2))--1x300x2 2x2
   model:add(ReLU(true))
   model:add(Dropout(0.4))
   model:add(Max(2,2,2,2))
   --
   model:add(Convolution(300,300,1,1))--1x300x1
   model:add(ReLU(true))
   model:add(Dropout(0.5))
   --
   model:add(Convolution(300,100,1,1))--1x100x1
   model:add(nn.Squeeze())

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(1/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')

   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
