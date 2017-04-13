local MyAdd, parent = torch.class('nn.MyAdd', 'nn.Module')

function MyAdd:__init(val)
   parent.__init(self)
  
   self.bias = torch.Tensor(1)
   self.gradBias = torch.Tensor(1)

   self:reset(val)
end

function MyAdd:reset(val)
   self.bias:fill(val)
end

function MyAdd:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:add(self.bias[1]);
   return self.output
end

function MyAdd:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
      return self.gradInput
   end
end

function MyAdd:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sum();
end