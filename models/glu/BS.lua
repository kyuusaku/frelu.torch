local BS, parent = torch.class('nn.BS', 'nn.Module')

function BS:__init(a, b)
   parent.__init(self)

   self.size = 1
   self.bias = torch.Tensor(self.size)
   self.gradBias = torch.Tensor(self.size)
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size)

   self:reset(a, b)
end

function BS:reset(a, b)
   self.bias:fill(a)
   self.weight:fill(b)
end

function BS:updateOutput(input)
   if self.weight[1] < 0 then
      self.weight[1] = 0.5
   end
   self.output:resizeAs(input):copy(input)
   self.output:add(self.bias[1])
   self.output:mul(self.weight[1])      
   return self.output
end

function BS:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:mul(self.weight[1])
   return self.gradInput
end

function BS:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sum()*self.weight[1]
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput) + scale*gradOutput:sum()*self.bias[1]
end