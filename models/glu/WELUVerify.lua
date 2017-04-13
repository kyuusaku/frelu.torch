require 'nn'
WELU=require 'WELU'

data = torch.randn(128,256,16,16)
mlp = nn.Sequential()
mlp:add(WELU(5,256))
print(mlp)
print(#mlp:forward(data))
