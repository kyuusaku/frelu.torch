require 'nn'
require './PSReLU'

local mytest = torch.TestSuite()

local tester = torch.Tester()

local jac = nn.Jacobian
local precision = 1e-5

function mytest.testPSReLU()
	
	local input = torch.rand(1, 10)
	local model = nn.PSReLU()
    local err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'PSReLU Linear gradient')

    input = torch.rand(5, 10, 6)
    model = nn.PSReLU()
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'PSReLU Map gradient')
    
    input = torch.rand(5, 10)
    model = nn.Sequential()
    model:add(nn.Linear(10, 10))
    model:add(nn.PSReLU(2))
    model:add(nn.LogSoftMax())
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'PSReLU with other layers')

end


tester:add(mytest)
tester:run()

