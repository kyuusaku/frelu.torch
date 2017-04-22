local MyAdd, parent = torch.class('nn.MyAdd', 'nn.Module')

function MyAdd:__init(val, inplace)
   parent.__init(self)
  
   self.bias = torch.Tensor(1)
   self.gradBias = torch.Tensor(1)
   self.inplace = inplace or false

   self:reset(val)
end

function MyAdd:reset(val)
   self.bias:fill(val)
end

function MyAdd:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   self.output:add(self.bias[1]);
   return self.output
end

function MyAdd:updateGradInput(input, gradOutput)
   if self.gradInput then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end 
      return self.gradInput
   end
end

function MyAdd:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sum();
end