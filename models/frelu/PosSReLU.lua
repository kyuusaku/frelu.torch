require 'nn'
require './MyAdd'

function PosSReLU(n, numInputDims, val, constant, inplace)
    n = n or 1
    val = val or -1
    constant = constant or false
    if inplace == nil then
        inplace = true
    end

    local m = nn.Sequential()
    m:add(nn.ReLU(true))
    if constant then
        -- https://github.com/torch/nn/blob/master/doc/transfer.md#addconstant
        m:add(nn.AddConstant(val, inplace))
    else
        m:add(nn.MyAdd(n, val, inplace, numInputDims))
    end

    return m
end

return PosSReLU
