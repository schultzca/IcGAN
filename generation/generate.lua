require 'image'
require 'nn'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

-- Load parameters from config file
assert(loadfile("cfg/generateConfig.lua"))(0)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
--torch.manualSeed(1)
assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
end

local ny -- Y label length. This depends on the dataset.
if opt.dataset == 'mnist' then ny = 10 
elseif opt.dataset == 'celebA' then ny = 18
else error('Not implemented.') end

assert(opt.batchSize >= ny, ('opt.batchSize must be equal or greater than %d (ny).'):format(ny))

if (opt.batchSize % ny) ~= 0 then
    local tmp = opt.batchSize + ny - opt.batchSize % ny
    print(('Warning: batchSize is not multiple of ny. Augmenting from %d to %d.'):format(opt.batchSize, tmp))
    opt.batchSize = tmp  
end

local Z = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
local Y = torch.zeros(opt.batchSize, ny):fill(-1)

-- Build Y
for i=1,opt.batchSize do
  Y[{{i},{((i-1)%ny)+1}}] = 1
end

local net = torch.load(opt.net)
net:evaluate()
if opt.noisetype == 'uniform' then
  -- Fix Z for every row in generated samples.
  -- A row has ny samples. Every i to (i-1)+ny samples
  -- will have the same Z.
  for i=1,opt.batchSize,ny do
      Z[{{i},{},{},{}}]:uniform(-1, 1)
      Z[{{i+1,(i-1)+ny},{},{},{}}] = Z[{{i},{},{},{}}]:expand(ny-1,opt.nz,opt.imsize,opt.imsize)
  end
elseif opt.noisetype == 'normal' then
  for i=1,opt.batchSize,ny do
      Z[{{i},{},{},{}}]:normal(0, 1)
      Z[{{i+1,(i-1)+ny},{},{},{}}] = Z[{{i},{},{},{}}]:expand(ny-1,opt.nz,opt.imsize,opt.imsize)
  end
end

if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    Z = Z:cuda()
    Y = Y:cuda()
else
   net:float()
end
local sampleInput = {Z:narrow(1,1,2), Y:narrow(1,1,2)}
-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(net, sampleInput)

local images = net:forward{Z, Y}
print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images,0,ny))
print('Saved image to: ', opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(image.toDisplayTensor(images,0,ny))
    print('Displayed image')
end
