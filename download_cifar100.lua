local opts = require 'opts'
local opt = opts.parse(arg)
local datasets = require 'datasets/init'
for i, split in ipairs{'train', 'val'} do 
  local dataset = datasets.create(opt, split)
end
