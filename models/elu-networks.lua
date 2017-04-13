local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ELU = nn.ELU

local function createModel(opt)
   local model = nn.Sequential()
   model:add(Convolution(3,192,5,5))--1x192x5
   --stack
   model:add(Convolution(192,192,1,1))--1x192x1
   model:add(Convolution(192,240,3,3))--1x240x3
   --stack
   model:add(Convolution(240,240,1,1))--1x240x1
   model:add(Convolution(240,260,2,2))--1x260x2
   --stack
   model:add(Convolution(260,260,1,1))--1x260x1
   model:add(Convolution(260,280,2,2))--1x280x2
   --stack
   model:add(Convolution(280,280,1,1))--1x280x1
   model:add(Convolution(280,300,2,2))--1x300x2
   --
   model:add(Convolution(300,300,1,1))--1x300x1
   --
   model:add(Convolution(300,100,1,1))--1x100x1

return createModel
