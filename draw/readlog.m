function [ data ] = readlog( filename )
%
%   
if ischar(filename)
    filenames{1}=filename;
else
    filenames=filename;
end
data=[];
for i=1:numel(filenames)
    filetemp=filenames{i};
    
    if ~exist(filetemp, 'file')
        error('cannot find the input file');
    end
    
    fidin=fopen(filetemp);              %打开文件

    while ~feof(fidin)                   %判断是不是文件末尾                                      
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

end

