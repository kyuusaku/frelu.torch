require 'torch'
require 'nn'
require 'gnuplot'
require './ShiftedReLU'

ii = torch.linspace(-3, 3)
m = ShiftedReLU(-1,false,false)
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.figure(1)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)
--gnuplot.axis({-3,3,-2,3})
--print(ii)
--print(oo)
--print(gi)

local mytest = torch.TestSuite()

local tester = torch.Tester()

local jac = nn.Jacobian
local precision = 1e-5

function mytest.testSReLU()
    
    local input = torch.rand(1, 10)
    local model = ShiftedReLU(-1,true,false)
    local err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(true) Linear gradient')

    input = torch.rand(5, 10, 6)
    model = ShiftedReLU(-1,true,false)
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(true) Map gradient')
    
    input = torch.rand(5, 10)
    model = nn.Sequential()
    model:add(nn.Linear(10, 10))
    model:add(ShiftedReLU(-1,true,false))
    model:add(nn.LogSoftMax())
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(true) with other layers')

    input = torch.rand(1, 10)
    model = ShiftedReLU(-1,false,false)
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(false) Linear gradient')

    input = torch.rand(5, 10, 6)
    model = ShiftedReLU(-1,false,false)
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(false) Map gradient')
    
    input = torch.rand(5, 10)
    model = nn.Sequential()
    model:add(nn.Linear(10, 10))
    model:add(ShiftedReLU(-1,false,false))
    model:add(nn.LogSoftMax())
    err = jac.testJacobian(model, input)
    tester:assert(err < precision, 'SReLU(false) with other layers')

end


tester:add(mytest)
tester:run()
