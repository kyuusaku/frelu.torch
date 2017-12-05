local SReLU, Parent = torch.class('nn.SReLU', 'nn.Module')

function SReLU:__init(val, inplace)
    parent.__init(self)
    
    self.size = 1;
    self.bias = torch.Tensor(self.size)
    self.gradBias = torch.Tensor(self.size)
    self.inplace = inplace or false

    self:reset(val)
end

function SReLU:reset(val)
    self.bias:fill(val)
end

function SReLU:updateOutput(input)
    if self.inplace then
        self.output = input
    else
        self.output:resizeAs(input):copy(input)
    end
    -- https://github.com/torch/torch7/blob/master/doc/maths.md#logical-operations-on-tensors
    self.output[torch.le(self.output, self.bias[1])] = self.bias[1]
    return self.output
end

function SReLU:updateGradInput(input, gradOutput)
    if self.gradInput then
        if self.inplace then
            self.gradInput = gradOutput
        else
            self.gradInput:resizeAs(gradOutput):copy(gradOutput)
        end
        self.gradInput[torch.le(input, self.bias[1])] = 0
        return self.gradInput
    end
end

function SReLU:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    self.gradBias[1] = self.gradBias[1] + scale*gradOutput[torch.le(input, self.bias[1])]:sum()
end