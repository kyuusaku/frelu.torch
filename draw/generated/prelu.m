function [ y ] = prelu( x ,k)
%PRELU �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
y=x.*(x>0)+k.*x.*(x<=0);

end

