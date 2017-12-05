require 'torch'
require 'nn'
require './ShiftedReLU'

ii = torch.linspace(-2, 2)
m = nn.ShiftedReLU()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)
