
%%
in = 'log/cifar100/';
out = 'dat/cifar100/';
network = 'pelu-smallnet';
log2dat(in, network, 'elu', out);
log2dat(in, network, 'elu-bn', out);
log2dat(in, network, 'relu', out);
log2dat(in, network, 'relu-bn', out);
log2dat(in, network, 'possrelu', out);
log2dat(in, network, 'possrelu-bn', out);
log2dat(in, network, 'pelu', out);

%%
showbestresult(out, network, 'elu');
showbestresult(out, network, 'elu-bn');
showbestresult(out, network, 'relu');
showbestresult(out, network, 'relu-bn');
showbestresult(out, network, 'possrelu');
showbestresult(out, network, 'possrelu-bn');
showbestresult(out, network, 'pelu');

%% test code
path = 'log/cifar100/';
network = 'pelu-smallnet';
act = 'elu';
data = cell(5,1);
for i = 1:5
    filename = [path network '-' act '-seed' num2str(i-1) '.log'];
    data{i} = readlog(filename);
end
train = zeros(size(data{1},1),5);
test = zeros(size(data{1},1),5);
for i = 1:5
    tmp = data{i};
    train(:,i) = tmp(:,2);
    test(:,i) = tmp(:,4);
end
csvwrite(['dat/cifar100/' network '-' act '-train.dat'], train);
csvwrite(['dat/cifar100/' network '-' act '-test.dat'], test);

%% test code
train = csvread(['dat/cifar100/' network '-' act '-train.dat']);
test = csvread(['dat/cifar100/' network '-' act '-test.dat']);
result.train = [mean(min(train)), std(min(train))];
result.test = [mean(min(test)), std(min(test))];
disp(result);

%% test code (correct)
path = 'log/';
network = 'pelu-smallnet';
act = 'elu';
filename = [path network '-' act '-seed' num2str(0) '.log'];
data = readlog(filename);
