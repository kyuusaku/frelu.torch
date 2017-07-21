function [ y ] = prelu( x ,k)
%PRELU 此处显示有关此函数的摘要
%   此处显示详细说明
y=x.*(x>0)+k.*x.*(x<=0);

end

