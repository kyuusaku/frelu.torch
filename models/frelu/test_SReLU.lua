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

