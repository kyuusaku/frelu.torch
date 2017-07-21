function [ y ] = pelu( x ,a, b)

y=(a/b)*x.*(x>0)+a*(exp(x/b)-1).*(x<=0);

end