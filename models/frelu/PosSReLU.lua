require 'nn'
require './MyAdd'

function PosSReLU(n, numInputDims, init_val, lr_decay_val, constant, inplace, isConstrained, constrainedVal)
    n = n or 1
    init_val = init_val or -1
    lr_decay_val = lr_decay_val or 1
    constant = constant or false
    if inplace == nil then
        inplace = true
    end
    isConstrained = isConstrained or false
    constrainedVal = constrainedVal or -0.1

    local m = nn.Sequential()
    m:add(nn.ReLU(true))
    if constant then
        -- https://github.com/torch/nn/blob/master/doc/transfer.md#addconstant
        m:add(nn.AddConstant(init_val, inplace))
    else
        m:add(nn.MyAdd(n, init_val, lr_decay_val, isConstrained, constrainedVal, inplace, numInputDims))
    end

    return m
end

return PosSReLU
