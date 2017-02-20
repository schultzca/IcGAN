require 'torch'
require 'nn'
require 'optim'

-- Load parameters from config file
assert(loadfile("cfg/mainConfig.lua"))(0)

-- one-line argument parser. Parses environment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
end
opt.manualSeed = torch.random(1, 10000) 
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
if opt.gpu > 0 then
    cutorch.manualSeed(opt.manualSeed)
end

if opt.saveGif > 0 then paths.mkdir('gif') end

opt.nThreads = 1 -- Do not change this parameter. This code can only use 1 thread for data handling.
torch.setnumthreads(opt.nThreads)
torch.setdefaulttensortype('torch.FloatTensor')
assert(opt.fineSize >= 8, "Minimum fineSize is 8x8.")
assert(opt.fineSize == 8 or opt.fineSize == 16 or opt.fineSize == 32 or opt.fineSize == 64 or opt.fineSize == 128 or opt.fineSize == 256 or opt.fineSize == 512 or opt.fineSize == 1024, "fineSize must be a power of 2.")

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

-- Replicate Y to match convolutional filter dimensions
local Yrepl = nn.Sequential()

-- ny -> ny x opt.fineSize/2 (replicate 2nd dimension)
Yrepl:add(nn.Replicate(opt.fineSize/2,2,1))
-- ny x 8 -> ny x opt.fineSize/2 x opt.fineSize/2 (replicate 3rd dimension)
Yrepl:add(nn.Replicate(opt.fineSize/2,3,2))


-- Join X and Y
pt:add(Xconv)
pt:add(Yrepl)
netD:add(pt)
netD:add(nn.JoinTable(1,3))

-- Convolutions applied on both X and Y
local inputFilters = ndf * fltMult + ny
for i=1,nConvLayers do
    -- state size: (ndf*fltMult + ny) x opt.fineSize/2^i x opt.fineSize/2^i
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
local Y_vis = torch.Tensor(opt.batchSize, ny):fill(-1)
local label = torch.Tensor(opt.batchSize) -- indicates whether images are real or generated
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   cutorch.setDevice(opt.gpu)
   X = X:cuda();  Z = Z:cuda(); Y = Y:cuda(); Y_vis = Y_vis:cuda(); label = label:cuda()

   if pcall(require, 'cudnn') then
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

if string.lower(opt.dataset) == 'celeba' then
    -- Use attributes from the training set to display higher quality samples.
    local _, yTmp , _ = data:getBatch()
    Y_vis:copy(yTmp[{{1,opt.batchSize},{}}])
end
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
   --local errD_real = torch.sum(output:lt(0.5))/output:size(1)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({X, Y}, df_do)
   
   -- Train with real images and wrong Y
   local errD_wrongY = 0
   if opt.trainWrongY then
      Y:copy(yWrong)
      label:fill(labelFake)
      
      output = netD:forward{X, Y}
      --errD_real = ((torch.sum(output:lt(0.5))/output:size(1)) + errD_real)/2
      errD_wrongY = 0.5*criterion:forward(output, label)
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
   --local errD_fake = torch.sum(output:ge(0.5))/output:size(1)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.trainWrongY then 
        df_do:mul(0.5) -- Fake image error is shared equally between real Y and wrong Y 
        errD_fake = errD_fake * 0.5
   end
   netD:backward({X, Y}, df_do)

   -- Error indicates % of how many samples have been incorrectly guessed by the discriminator
   errD = (errD_real + errD_fake + errD_wrongY) / 2

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
    labels = {'Epoch', 'G error', 'D error'},
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
local nSamples = math.min(data:size(), opt.ntrain)
local batchIterations = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, nSamples, opt.batchSize do
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
          disp.image(image.toDisplayTensor(fake,0,ny), {win=opt.display_id, title=opt.name .. '. Generated images'})
          disp.image(real, {win=opt.display_id*3, title=opt.name .. '. Real images'})
          -- display generator and discriminator error
          table.insert(errorData,
          {
            batchIterations/math.ceil(nSamples / opt.batchSize), -- x-axis
            errG, -- y-axis for label1
            errD -- y-axis for label2
          })
          disp.plot(errorData, errorDispConfig)
          if opt.saveGif > 0 then 
            image.save(('gif/%.4f.jpg'):format(batchIterations/math.ceil(nSamples / opt.batchSize)), 
                       image.toDisplayTensor(fake,0,torch.round(math.sqrt(opt.batchSize)),true)) 
          end
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
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   if opt.display then
      torch.save('checkpoints/' .. opt.name .. '_error.t7', errorData)
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

if opt.poweroff > 0 then os.execute("poweroff") end
