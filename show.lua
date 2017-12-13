require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
PosSReLU = require './models/frelu/PosSReLU'



local function parse(arg)
	local cmd = torch.CmdLine()
	cmd:option('-model',      'none',   'Path to model to show')
	local opt = cmd:parse(arg or {})
	return opt
end

local function show(model, name)   
    for k,v in pairs(model:findModules(name)) do
        if v.weight ~= nil then
            print('weight:', v.weight)
        end
        if v.bias ~= nil then
            print('bias:', v.bias)
        end
    end
end

local opt = parse(arg)

local model = torch.load(opt.model)

print(model)

show(model,'nn.MyAdd')