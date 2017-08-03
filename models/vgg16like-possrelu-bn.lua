--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local PosSReLU = require './glu/PosSReLU'
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local Dropout = nn.Dropout

local function createModel(opt)
   local function Block(...)
      local arg = {...}
      return nn.Sequential()
         :add(Convolution(...))
         :add(SBatchNorm(arg[2]))
         :add(PosSReLU(1))
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for imagenet:
      -- http://ethereon.github.io/netscope/#/preset/nin      
      print(' | IN ImageNet')
      -- The NIN ImageNet model
      model:add(Block(3,96,11,11,4,4)) -- conv1
      model:add(Block(96,96,1,1)) -- cccp1
      model:add(Block(96,96,1,1)) -- cccp2
      model:add(Max(3,3,2,2)) -- pool1
      model:add(Block(96,256,5,5,1,1,2,2)) -- conv2
      model:add(Block(256,256,1,1)) -- cccp3
      model:add(Block(256,256,1,1)) -- cccp4
      model:add(Avg(3,3,2,2)) -- pool2
      model:add(Block(256,384,3,3,1,1,1,1)) -- conv3
      model:add(Block(384,384,1,1)) -- cccp5
      model:add(Block(384,384,1,1)) -- cccp6
      model:add(Max(3,3,2,2)) -- pool3
      model:add(Dropout(0.5)) -- drop
      model:add(Block(384,1024,3,3,1,1,1,1)) -- conv4-1024
      model:add(Block(1024,1024,1,1)) -- cccp7-1024
      model:add(Block(1024,1000,1,1)) -- cccp8-1024
      model:add(Avg(6,6,1,1)) -- pool4
      model:add(nn.View(1000):setNumInputDims(3))
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      -- https://gist.github.com/mavenlin/e56253735ef32c3c296d
      print(' | vgg16like CIFAR-10')
      -- The NIN CIFAR-10 model
      model:add(Block(3,64,3,3,1,1,1,1))
      model:add(Block(64,64,3,3,1,1,1,1)) -- 32x32
      model:add(Max(2,2,2,2))
      model:add(Block(64,128,3,3,1,1,1,1))
      model:add(Block(128,128,3,3,1,1,1,1)) -- 16x16
      model:add(Max(2,2,2,2))
      model:add(Block(128,256,3,3,1,1,1,1))
      model:add(Block(256,256,3,3,1,1,1,1))
      model:add(Block(256,256,3,3,1,1,1,1)) -- 8x8
      model:add(Max(2,2,2,2)) -- 4x4
      model:add(Block(256,2048,4,4,1,1,0,0)) -- 1x1
      model:add(Dropout(0.5))
      model:add(Block(2048,512,1,1,1,1,0,0)) -- 1x1
      model:add(Dropout(0.5))
      model:add(nn.View(-1):setNumInputDims(3))
      model:add(nn.Linear(512,10)) 
      -- model:add(nn.BatchNormalization(10))     
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      -- https://github.com/mavenlin/cuda-convnet/blob/master/NIN/cifar-100_def
      print(' | vgg16like CIFAR-100')
      -- The NIN CIFAR-100 model
      model:add(Block(3,64,3,3,1,1,1,1))
      model:add(Block(64,64,3,3,1,1,1,1)) -- 32x32
      model:add(Max(2,2,2,2))
      model:add(Block(64,128,3,3,1,1,1,1))
      model:add(Block(128,128,3,3,1,1,1,1)) -- 16x16
      model:add(Max(2,2,2,2))
      model:add(Block(128,256,3,3,1,1,1,1))
      model:add(Block(256,256,3,3,1,1,1,1))
      model:add(Block(256,256,3,3,1,1,1,1)) -- 8x8
      model:add(Max(2,2,2,2)) -- 4x4
      model:add(Block(256,2048,4,4,1,1,0,0)) -- 1x1
      model:add(Dropout(0.5))
      model:add(Block(2048,512,1,1,1,1,0,0)) -- 1x1
      model:add(Dropout(0.5))
      model:add(nn.View(-1):setNumInputDims(3))
      model:add(nn.Linear(512,100))
      -- model:add(nn.BatchNormalization(100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

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
