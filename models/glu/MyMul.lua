local MyMul, parent = torch.class('nn.MyMul', 'nn.Module')

function MyMul:__init(val)
   parent.__init(self)

   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)

   self:reset(val)
end


function MyMul:reset(val)
   self.weight:fill(val);
end

function MyMul:updateOutput(input)
   self.output:resizeAs(input):copy(input);
   self.output:mul(self.weight[1]);
   return self.output
end

function MyMul:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add(self.weight[1], gradOutput)
   return self.gradInput
end

function MyMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
end