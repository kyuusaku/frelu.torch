local MyMul, parent = torch.class('nn.MyMul', 'nn.Module')

function MyMul:__init(val, inplace)
   parent.__init(self)

   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   self.inplace = inplace or false

   self:reset(val)
end


function MyMul:reset(val)
   self.weight:fill(val);
end

function MyMul:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input);
   end
   self.output:mul(self.weight[1]);
   return self.output
end

function MyMul:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput = gradOutput
      self.gradInput:mul(self.weight[1])
   else
      self.gradInput:resizeAs(input):zero()
      self.gradInput:add(self.weight[1], gradOutput)
   end
   return self.gradInput
end

function MyMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
end