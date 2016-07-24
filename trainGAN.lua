require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'mnist',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 32, -- 96,
   fineSize = 32, -- 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in last deconv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'c_mnist',
   noise = 'normal',       -- uniform / normal
   dataRoot = 'mnist',
   -- Parameters for conditioned GAN
   trainWrongY = true   -- explicitly train discriminator with real images and wrong Y error
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
assert(opt.fineSize >= 8, "Minimum fineSize is 8x8.")
assert(opt.fineSize % 2 == 0, "fineSize must be multiple of 2.")

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
if opt.dataset == 'mnist' then
  nc = 1
end
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local ny = data:ySize()
local labelReal = 1
local labelFake = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

-- --==GENERATOR==--

local netG = nn.Sequential()
-- Concatenate Z and Y
--    Concatenate on first (non-batch) dimension, 
--    where non-batch input has 3 dimensions
netG:add(nn.JoinTable(1,3))
 
-- Calculate number of deconv layers given the output image size
-- First and last layers are not included, as they have a different configuration
local nConvLayers = math.log(opt.fineSize,2) - 3
-- Filter multiplier used to decrement the # of filter for every deconv layer
-- For every new layer the multiplier is divided by two.
local fltMult = opt.fineSize / 8

-- input is Z+Y, going into a convolution
netG:add(SpatialFullConvolution(nz + ny, ngf * fltMult, 4, 4))
netG:add(SpatialBatchNormalization(ngf * fltMult)):add(nn.ReLU(true))

for i=1,nConvLayers do
   --- state size: (ngf * fltMult) x 2^(i+1) x 2^(i+1)
    netG:add(SpatialFullConvolution(ngf * fltMult, ngf * fltMult/2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * fltMult/2)):add(nn.ReLU(true))
    fltMult = fltMult / 2
end

-- state size: ngf x opt.fineSize/2 x opt.fineSize/2
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: nc x opt.fineSize x opt.fineSize

netG:apply(weights_init)

-- --==DISCRIMINATOR==--

local netD = nn.Sequential()

-- Need a parallel table to put different layers for X (conv layers) 
-- and Y (none) before joining both inputs together.
local pt = nn.ParallelTable()

-- Convolutions applied only on X input
local Xconv = nn.Sequential() 
-- input is nc x opt.fineSize x opt.fineSize
Xconv:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
Xconv:add(nn.LeakyReLU(0.2, true))

fltMult = 1

if nConvLayers > 0 then -- if input images are greater than 8x8
    -- state size: ndf x opt.fineSize/2 x opt.fineSize/2
    fltMult = 2
    Xconv:add(SpatialConvolution(ndf, ndf * fltMult, 4, 4, 2, 2, 1, 1))
    Xconv:add(SpatialBatchNormalization(ndf * fltMult)):add(nn.LeakyReLU(0.2, true))
end

-- Replicate Y to match convolutional filter dimensions
local Yrepl = nn.Sequential()

if nConvLayers == 0 then
    -- ny -> ny x 4 (replicate 2nd dimension)
    Yrepl:add(nn.Replicate(4,2,1)) -- 8-> MNIST (size 32x32), 16 -> celebA (size 64x64)
    -- ny x 8 -> ny x 4 x 4 (replicate 3rd dimension)
    Yrepl:add(nn.Replicate(4,3,2))
else
    -- ny -> ny x opt.fineSize/4 (replicate 2nd dimension)
    Yrepl:add(nn.Replicate(opt.fineSize/4,2,1)) -- 8-> MNIST (size 32x32), 16 -> celebA (size 64x64)
    -- ny x 8 -> ny x opt.fineSize/4 x opt.fineSize/4 (replicate 3rd dimension)
    Yrepl:add(nn.Replicate(opt.fineSize/4,3,2))
end

-- Join X and Y
pt:add(Xconv)
pt:add(Yrepl)
netD:add(pt)
netD:add(nn.JoinTable(1,3))

-- Convolutions applied on both X and Y
local inputFilters = ndf * fltMult + ny
for i=2,nConvLayers do -- starts with 2 because one conv layer was already introduced before
    -- state size: (ndf*fltMult + ny) x opt.fineSize/2^i x opt.fineSize2^i
    
    netD:add(SpatialConvolution(inputFilters, ndf * fltMult * 2, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * fltMult * 2)):add(nn.LeakyReLU(0.2, true))

    fltMult = fltMult * 2
    inputFilters =  ndf * fltMult
end

-- state size: (ndf*fltMult) x 4 x 4
netD:add(SpatialConvolution(inputFilters, 1, 4, 4))

netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(nc))
-- state size: 1

netD:apply(weights_init)

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local X = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize) -- input images
local Z = torch.Tensor(opt.batchSize, nz, 1, 1) -- input noise
local Y = torch.Tensor(opt.batchSize, ny)
local Y_vis = torch.zeros(opt.batchSize, ny)
local label = torch.Tensor(opt.batchSize) -- indicates whether images are real or generated
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   X = X:cuda();  Z = Z:cuda(); Y = Y:cuda(); Y_vis = Y_vis:cuda(); label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local noise_vis = Z:clone()
-- PROVISIONAL --
for i=1,opt.batchSize do
  Y_vis[{{i},{((i-1)%ny)+1}}] = 1
end

if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local xReal, yReal, yWrong = data:getBatch()
   data_tm:stop()
   X:copy(xReal)
   Y:copy(yReal)
   label:fill(labelReal)

   -- Train with real images X and correct conditioning vectors Y
   local output = netD:forward{X, Y}
   local errD_real = torch.sum(output:lt(0.5))/output:size(1)
   criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.trainWrongY then 
        df_do:mul(0.5) -- Real image error is shared equally between real Y and wrong Y 
   end
   netD:backward({X, Y}, df_do)
   
   -- Train with real images and wrong Y
   if opt.trainWrongY then
      Y:copy(yWrong)
      label:fill(labelFake)
      
      output = netD:forward{X, Y}
      --errD_real = ((torch.sum(output:lt(0.5))/output:size(1)) + errD_real)/2
      criterion:forward(output, label)
      df_do = criterion:backward(output, label)
      df_do:mul(0.5)
      netD:backward({X, Y}, df_do)
   end

   -- Train with fake images and sampled Y
   if opt.noise == 'uniform' then -- regenerate random noise
       Z:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       Z:normal(0, 1)
   end
   
   local yFake = data:sampleY()
   Y:copy(yFake)
   local xFake = netG:forward{Z, Y}
   X:copy(xFake)
   label:fill(labelFake)

   local output = netD:forward{X, Y}
   local errD_fake = torch.sum(output:ge(0.5))/output:size(1)
   criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({X, Y}, df_do)

   -- Error indicates % of how many samples have been incorrectly guessed by the discriminator
   errD = (errD_real + errD_fake) / 2

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   Z:uniform(-1, 1) -- regenerate random noise
   local xFake = netG:forward(Z)
   X:copy(xFake) 
   Y:copy(yFake)]]--
   label:fill(labelReal) -- fake labels are real for generator cost
  
   -- Need to compute output again, as D has been updated in fDx
   local output = netD:forward{X, Y}
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput({X, Y}, df_do)
   -- df_dg contains the gradients w.r.t. generated images by G,
   -- which is precisely what you want to apply at G's backward step.
   -- df_do is not used on G's bkwrd step because they are the gradients
   -- w.r.t. D output, which is only a one-dimensional vector with
   -- the score between 0 and 1 of real/fake of a set of generated images.
   
   -- df_dg[1] contains the gradient of generated images X (G output, D input)
   -- which are the ones you need to compute the backward step.
   -- df_dg[2] contains the gradient of Y. You don't need them as they are not G's output.
   netG:backward({Z, Y}, df_dg[1])  
   return errG, gradParametersG
end

-- initialize generator and discriminator error display configuration
local errorData
local errorDispConfig
if opt.display then
  errorData = {}
  errorDispConfig =
  {
    title = 'Generator and discriminator error',
    win = opt.display_id * 4,
    labels = {'Batch iterations', 'G error', 'D error'},
    ylabel = "G error",
    y2label = "D error",
    legend = 'always',
    axes = { y2 = {valueRange = {0,1}}},
    series = {
      ['D error'] = { axis = 'y2' }
    }
  }
end

-- train
local batchIterations = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)
      
      batchIterations = batchIterations + 1
      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward{noise_vis, Y_vis}
          local real = data:getBatch()
          disp.image(image.toDisplayTensor(fake,0,10), {win=opt.display_id, title=opt.name .. '. Generated images'})
          disp.image(real, {win=opt.display_id*3, title=opt.name .. '. Real images'})
          -- display generator and discriminator error
          table.insert(errorData,
          {
            batchIterations, -- x-axis
            errG, -- y-axis for label1
            errD -- y-axis for label2
          })
          disp.plot(errorData, errorDispConfig)
          
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
