require 'nn'
require './GLU'

local mytest = torch.TestSuite()

local tester = torch.Tester()

local jac = nn.Jacobian
local precision = 1e-5

function mytest.testGLU()
	
	local input = torch.rand(1, 10)
	for i=1,3 do
		local model = nn.GLU(i)
        local err = jac.testJacobian(model, input)
        tester:assert(err < precision, string.format('GLU(%d) Linear gradient', i))
    end

    input = torch.rand(5, 10, 6)
    for i=1,3 do
		local model = nn.GLU(i)
        local err = jac.testJacobian(model, input)
        tester:assert(err < precision, string.format('GLU(%d) Map gradient', i))
    end
    
    input = torch.rand(5, 10)
    local model = nn.Sequential()
    model:add(nn.Linear(10, 10))
    model:add(nn.GLU(2))
    model:add(nn.LogSoftMax())
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'GLU(2) with other layers')

end


tester:add(mytest)
tester:run()

