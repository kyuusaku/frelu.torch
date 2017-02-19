local GLU, parent = torch.class('nn.GLU', 'nn.Module')

function GLU:__init(num_linear)
	parent.__init(self)

    self.num_linear = num_linear or 2
	self.weight = torch.Tensor(self.num_linear)
	self.gradWeight = torch.Tensor(self.num_linear)
	self.bias = torch.Tensor(self.num_linear)
	self.gradBias = torch.Tensor(self.num_linear)

	self:reset()
end

function GLU:reset()
	local step = 1.0/self.num_linear
	for i=1,self.num_linear do
		self.weight:uniform((i-1)*step,i*step)
	end
	self.bias:zero()
end

local function compute_linear(self, input)
	local input_vector = input:view(-1,1)
	local scale_input = input_vector * self.weight:view(1,-1) 
	                    + torch.expand(self.bias:view(1,-1), input_vector:size(1), self.num_linear)
	return scale_input
end

local function compute_weight(scale_input)
	local exp_input = torch.exp(scale_input - torch.max(scale_input))
    local weight_input = torch.cdiv(exp_input, 
    	torch.expand( torch.sum(exp_input,2), exp_input:size(1), exp_input:size(2) ) )
    return weight_input
end

local function compute_gradWeightInput(self, weight_input)
	local eq = torch.cmul(weight_input, 
		torch.expand(self.weight:view(1,-1), weight_input:size(1), self.num_linear) )
	local grad_weight_input = eq:clone()
	for i=1,self.num_linear do
		local a = weight_input:select(2,i):clone():view(-1,1)
		grad_weight_input:select(2,i):add(-torch.sum(torch.cmul(eq,torch.expand(a,eq:size(1),eq:size(2))), 2))
	end
	return grad_weight_input
end

local function compute_gradWeight(self, input, scale_input, weight_input)
	local input_vector = input:view(-1,1)
	local eq = torch.cmul(weight_input, 
                torch.expand(input_vector, weight_input:size(1), weight_input:size(2)) )
	local grad_weight = eq:clone()
        for i=1,self.num_linear do
    	    local a = weight_input:select(2,i):clone():view(-1,1)
		grad_weight:select(2,i):add(-torch.sum(torch.cmul(eq,torch.expand(a,eq:size(1),eq:size(2))), 2))
	end
	return grad_weight:cmul(scale_input) + eq
end

local function compute_gradBias(self, scale_input, weight_input)
	local grad_bias = weight_input:clone()
    for i=1,self.num_linear do
    	local a = weight_input:select(2,i):clone():view(-1,1)
		grad_bias:select(2,i):add(-torch.sum(torch.cmul(weight_input,
			torch.expand(a,weight_input:size(1),weight_input:size(2))), 2))
	end
	return grad_bias:cmul(scale_input) + weight_input
end

function GLU:updateOutput(input)
	local scale_input = compute_linear(self, input)
    local weight_input = compute_weight(scale_input)    
	self.output = torch.sum(torch.cmul(scale_input, weight_input),2)
	self.output = self.output:viewAs(input)
	return self.output
end

function GLU:updateGradInput(input, gradOutput)
    local scale_input = compute_linear(self, input)
    local weight_input = compute_weight(scale_input)
    local grad_weight_input = compute_gradWeightInput(self, weight_input)
    self.gradInput = torch.add( torch.cmul(grad_weight_input,scale_input), torch.cmul(weight_input,
    	torch.expand(self.weight:view(1,-1), weight_input:size(1), weight_input:size(2))) )
    self.gradInput = torch.sum(self.gradInput, 2)
    self.gradInput = torch.cmul(self.gradInput:viewAs(input), gradOutput)
    return self.gradInput
end

function GLU:accGradParameters(input, gradOutput, scale)
	local grad_output_vector = gradOutput:view(-1,1)
	local scale_input = compute_linear(self, input)
    local weight_input = compute_weight(scale_input)
    local grad_weight = compute_gradWeight(self, input, scale_input, weight_input)
    local grad_bias = compute_gradBias(self, scale_input, weight_input)
    grad_weight:cmul(torch.expand(grad_output_vector, grad_weight:size(1), grad_weight:size(2)))
    grad_bias:cmul(torch.expand(grad_output_vector, grad_bias:size(1), grad_bias:size(2)))
    scale = scale or 1
    self.gradWeight = self.gradWeight + scale*torch.sum(grad_weight,1)
    self.gradBias = self.gradBias + scale*torch.sum(grad_bias,1)
end
