require 'image'
require 'nn'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
    batchSize = 100,        -- number of samples to produce (it should be multiple of ny)
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,              
    -- Conditioned GAN parameters
    dataset = 'mnist'     -- mnist | celebA           
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
end

local ny -- Y label length. This depends on the dataset.
if opt.dataset == 'mnist' then ny = 10 
elseif opt.dataset == 'celebA' then ny = nil error('Not implemented.') end

assert(opt.batchSize >= ny, ('opt.batchSize must be equal or greater than %d (ny).'):format(ny))

if (opt.batchSize % ny) ~= 0 then
    local tmp = opt.batchSize + ny - opt.batchSize % ny
    print(('Warning: batchSize is not multiple of ny. Augmenting from %d to %d.'):format(opt.batchSize, tmp))
    opt.batchSize = tmp  
end

local Z = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
local Y = torch.zeros(opt.batchSize, ny)

-- Y is specific for MNIST dataset
if opt.dataset == 'mnist' then
    for i=1,opt.batchSize do
      Y[{{i},{((i-1)%ny)+1}}] = 1
    end
elseif opt.dataset == 'celebA' then
    error('Not implemented.')
end

local net = torch.load(opt.net)

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

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        Z:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    Z = Z:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        Z:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        Z:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end

local sample_input = {torch.randn(2,opt.nz,1,1), torch.randn(2,ny)}
if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    Z = Z:cuda()
    Y = Y:cuda()
    sample_input[1] = sample_input[1]:cuda()
    sample_input[2] = sample_input[2]:cuda()
else
   sample_input[1] = sample_input[1]:float()
   sample_input[2] = sample_input[2]:float()
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
