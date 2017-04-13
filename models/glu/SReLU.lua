local SReLU, Parent = torch.class('nn.SReLU', 'nn.Threshold')

function SReLU:__init(b,ip)
    Parent.__init(self,b,b,ip)
end