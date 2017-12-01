local MyAdd, parent = torch.class('nn.MyAdd', 'nn.Module')

function MyAdd:__init(n, val, inplace, numInputDims)
   parent.__init(self)

   self.size = n;  
   self.bias = torch.Tensor(self.size)
   self.gradBias = torch.Tensor(self.size)
   self.inplace = inplace or false
   self:setNumInputDims(numInputDims);

   self:reset(val)
end

function MyAdd:reset(val)
   self.bias:fill(val)
end

function MyAdd:setNumInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

function MyAdd:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.size == 1 then
      self.output:add(self.bias[1]);
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
            self.output[i]:add(self.bias[i])
         end
      else
         assert(self.size == input:size(2),
            string.format('got %dD channels, %dD self.size', input:size(2), self.size))
         for i = 1, self.size do
            self.output[{{},i}]:add(self.bias[i])
         end
      end
   end
   return self.output
end

function MyAdd:updateGradInput(input, gradOutput)
   if self.gradInput then
      if self.inplace then
         self.gradInput = gradOutput
         -- restore previous input value
         if self.size == 1 then
            input:add(-self.bias[1])
         else
            local iDim = input:dim()
            if self.numInputDims == iDim then
               for i = 1, self.size do
                  input[i]:add(-self.bias[i])
               end
            else
               for i = 1, self.size do
                  input[{{},i}]:add(-self.bias[i])
               end
            end
         end
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end 
      return self.gradInput
   end
end

function MyAdd:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.size == 1 then
      self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sum();
   else
      local iDim = input:dim()
      if self.numInputDims == iDim then
         for i = 1, self.size do
            self.gradBias[i] = self.gradBias[i] + scale*gradOutput[i]:sum();
         end
      else
         for i = 1, self.size do
            self.gradBias[i] = self.gradBias[i] + scale*gradOutput[{{},i}]:sum();
         end
      end
   end
end