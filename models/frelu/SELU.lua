require 'nn'

function SELU(inplace)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return nn.Sequential()
        :add(nn.ELU(alpha, inplace))
        :add(nn.MulConstant(scale, true))
end

return SELU
