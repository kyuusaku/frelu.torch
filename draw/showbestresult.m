function showbestresult(path, network, act)

train = csvread([path network '-' act '-train.dat']);
test = csvread([path network '-' act '-test.dat']);
result.train = [mean(min(train)), std(min(train))];
result.test = [mean(min(test)), std(min(test))];
disp(result);