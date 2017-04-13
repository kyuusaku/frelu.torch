local PSReLU, parent = torch.class('nn.PSReLU','nn.Module')

function PSReLU:__init()
   parent.__init(self)
   -- cannot inplace
   self.bias = torch.Tensor(1):fill(-1)
   self.gradBias = torch.Tensor(1)
end

function PSReLU:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output[input:le(self.bias[1])] = self.bias[1]
   return self.output
end

function PSReLU:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput[input:le(self.bias[1])] = 0
   return self.gradInput
end

function PSReLU:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradBias[1] = self.gradBias[1] + scale*gradOutput[input:le(self.bias[1])]:sum()
end


