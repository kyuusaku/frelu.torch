function [ y ] = elu( x )
%ELU �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
y=x.*(x>0)+(exp(x)-1).*(x<=0);

end

