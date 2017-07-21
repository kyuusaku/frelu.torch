function log2dat(in, network, act, out)


data = cell(5,1);
for i = 1:5
    filename = [in network '-' act '-seed' num2str(i-1) '.log'];
    data{i} = readlog(filename);
end
train = zeros(size(data{1},1),5);
test = zeros(size(data{1},1),5);
for i = 1:5
    tmp = data{i};
    train(:,i) = tmp(:,2);
    test(:,i) = tmp(:,4);
end
csvwrite([out network '-' act '-train.dat'], train);
csvwrite([out network '-' act '-test.dat'], test);