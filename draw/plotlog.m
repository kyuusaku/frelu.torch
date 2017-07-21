function [ data ] = plotlog( filename, color)
%PLOTLOG �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if ischar(filename)
    filenames{1}=filename;
else
    filenames=filename;
end
data=[];
for i=1:numel(filenames)
    filetemp=filenames{i};
    fidin=fopen(filetemp);              %���ļ�

    while ~feof(fidin)                   %�ж��ǲ����ļ�ĩβ                                      
       tline=fgetl(fidin);
       if length(tline)>=2
           if tline(2)=='*'
               str_data = regexp( tline, '([\d|\.]+)',  'match' );
               if length(str_data)==9
                   temp(1)=str2double(str_data{1});
                   temp(2)=str2double(str_data{3});
                   temp(3)=str2double(str_data{5});
                   temp(4)=str2double(str_data{7});
                   temp(5)=str2double(str_data{9});
                   data=[data;temp];
               end
           end
       end
    end
    fclose(fidin);
end
train_str=['Train[' num2str(min(data(:,2))) ']'];
test_str=['Test [' num2str(min(data(:,4))) ']'];
plot(data(:,1),data(:,2),color,'LineWidth',1,'LineStyle','--','DisplayName',train_str);
plot(data(:,1),data(:,4),color,'LineWidth',2,'LineStyle','-','DisplayName',test_str);

end

