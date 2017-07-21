local MyConstrainedMul, parent = torch.class('nn.MyConstrainedMul', 'nn.Module')

function MyConstrainedMul:__init(v)
   parent.__init(self)
   
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)

   self:reset(v)
end


function MyConstrainedMul:reset(v)
   self.weight:fill(v)
end

function MyConstrainedMul:updateOutput(input)
   print(('%.5f'):format(self.weight[1]))
   if self.weight[1] < 0.1 then
      self.weight[1] = 0.1
   end
   self.output:resizeAs(input):copy(input);
   self.output:mul(self.weight[1]);
   return self.output
end

function MyConstrainedMul:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add(self.weight[1], gradOutput)
   return self.gradInput
end

function MyConstrainedMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
end
