require 'nn'
require './SReLU'

function ShiftedReLU(val, inplace, constant)
    val = val or -1
    constant = constant or false
    if inplace == nil then
        inplace = true
    end

    local m = nn.Sequential()
    if constant then
        m:add(nn.Threshold(val,val,inplace))
    else
        m:add(nn.SReLU(val, inplace))
    end

    return m
end

return ShiftedReLU
