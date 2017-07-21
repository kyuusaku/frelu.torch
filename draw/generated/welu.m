function [y, l1, l2] = welu(x,a1,b1,a2,b2)

l1=a1*x+b1;
l2=a2*x+b2;
y=(exp(l1).*l1 + exp(l2).*l2) ./ (exp(l1) + exp(l2));