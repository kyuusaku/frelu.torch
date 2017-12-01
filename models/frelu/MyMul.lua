local MyMul, parent = torch.class('nn.MyMul', 'nn.Module')

function MyMul:__init(n, val, inplace, numInputDims)
   parent.__init(self)

   self.size = n;
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size)
   self.inplace = inplace or false
   self:setNumInputDims(numInputDims);

   self:reset(val)
end


function MyMul:reset(val)
   self.weight:fill(val);
end

function MyMul:setNumInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

function MyMul:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input);
   end
   if self.size == 1 then
      self.output:mul(self.weight[1]);
   else
      local iDim = input:dim()
      if self.numInputDims ~= nil then
         assert(self.numInputDims == iDim or self.numInputDims == iDim - 1,
            string.format('got %dD input, %dD numInputDims', iDim, self.numInputDims))
      end
      if self.numInputDims == iDim then
         assert(self.size == input:size(1),
            string.format('got %dD channels, %dD self.size', input:size(1), self.size))
         for i = 1, self.size do
            self.output[i]:mul(self.weight[i])
         end
      else
         assert(self.size == input:size(2),
            string.format('got %dD channels, %dD self.size', input:size(2), self.size))
         for i = 1, self.size do
            self.output[{{},i}]:mul(self.weight[i])
         end
      end
   end
   return self.output
end

function MyMul:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput = gradOutput
      if self.size == 1 then
         self.gradInput:mul(self.weight[1])
      else
         local iDim = input:dim()
         if self.numInputDims == iDim then
            for i = 1, self.size do
               self.gradInput[i]:mul(self.weight[i])
            end
         else
            for i = 1, self.size do
               self.gradInput[{{},i}]:mul(self.weight[i])
            end
         end
      end
   else
      self.gradInput:resizeAs(input):zero()
      if self.size == 1 then
         self.gradInput:add(self.weight[1], gradOutput)
      else
         local iDim = input:dim()
         if self.numInputDims == iDim then
            for i = 1, self.size do
               self.gradInput[i]:add(self.weight[i], gradOutput[i])
            end
         else
            for i = 1, self.size do
               self.gradInput[{{},i}]:add(self.weight[i], gradOutput[{{},i}])
            end
         end
      end
   end
   return self.gradInput
end

function MyMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.size == 1 then
      self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
   else
      local iDim = input:dim()
      if self.numInputDims == iDim then
         for i = 1, self.size do
            self.gradWeight[i] = self.gradWeight[i] + scale*input[i]:dot(gradOutput[i]);
         end
      else
         for i = 1, self.size do
            self.gradWeight[i] = self.gradWeight[i] + scale*input[{{},i}]:dot(gradOutput[{{},i}]);
         end
      end
   end
end