function [ y ] = elu( x )
%ELU 此处显示有关此函数的摘要
%   此处显示详细说明
y=x.*(x>0)+(exp(x)-1).*(x<=0);

end

