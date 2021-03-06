local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local SBatchNorm = nn.SpatialBatchNormalization
local Max = nn.SpatialMaxPooling
local PReLU = nn.PReLU
local Dropout = nn.Dropout

local function convblock(ninput, noutput)
   return nn.Sequential()
      :add(Convolution(ninput,noutput,3,3,1,1,1,1))
      :add(SBatchNorm(noutput))
      :add(PReLU())
      :add(Max(2,2,2,2))
      :add(Dropout(0.2))
end

local function createModel(opt)
   local model = nn.Sequential()
   model:add(convblock(3,32)) -- 16x16
   model:add(convblock(32,64)) -- 8x8
   model:add(convblock(64,128)) -- 4x4
   model:add(nn.View(-1):setNumInputDims(3))
   model:add(nn.Linear(2048,512))
   model:add(nn.BatchNormalization(512))
   model:add(PReLU())
   model:add(Dropout(0.5))
   model:add(nn.Linear(512,100))
   
   local function ConvInit(name)   
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   BNInit('nn.BatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
