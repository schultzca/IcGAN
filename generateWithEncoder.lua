require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
    nImages = 13,         -- number of samples to produce (only valid if loadOption != 1)
    decNet = 'checkpoints/c_mnist_25_net_G.t7',--'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    encNet = 'checkpoints/encoder_c_mnist_6epochs.t7',--'checkpoints/encoder128Filters2FC_dataset2_2_6epochs.t7',
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    loadOption = 2,  -- 0 = only generated images used, 1 = load input image, 2 = load multiple input images
    loadPath = 'mnist/imagesTest', --'mnist/images', -- path used when load is 1 (path to single image) or 2 (path to folder with images)
    name = 'encoder_disentangle',
    -- Conditional GAN parameters
    dataset = 'mnist',
}

local function sampleY(Y, dataset, ny, batchSize)
   if dataset == 'mnist' then
      for i=1,batchSize do
          Y[{{i},{((i-1)%ny)+1}}] = 1
      end  
  elseif dataset == 'celebA' then
      error('Not implemented.')
  else
      error('Dataset '..dataset..'not recognized.')
  end
end 

local function obtainImageSet(X, path, option, extension)
    if option == 1 then
        -- Load input image X
        -- Check string is a path to an image
        assert(path:match(extension) ~= nil, "opt.loadPath '"..path.."' is not an image.")
        
        local tmp = image.load(path):float()
        tmp = image.scale(tmp, X:size(3), X:size(4))
        -- Image dimensions is 3. We need a 4th dimension indicating the number of images.
        tmp:resize(1, X:size(2), X:size(3), X:size(4))
        X = tmp
    elseif option == 2 then
        -- Load multiple images given a path
        assert(paths.dir(path)~=nil, "opt.loadPath '"..path.."' is not a directory.")
        local i = 1
        local fileIterator = paths.files(path, extension)
        while i <= opt.nImages do
            local imPath = path .. '/' .. fileIterator()
            local im = image.load(imPath)
            X[{{i},{},{},{}}] = image.scale(im, X:size(3), X:size(4))
            i = i + 1
        end
    else
        error('Option (customInputImage) not recognized.')
    end
    X:mul(2):add(-1) -- change [0, 1] to [-1, 1]
    return X
end


if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

if opt.loadOption == 1 then opt.nImages = 1 end

local imgExtension = '.png'
local ny -- Y label length. This depends on the dataset.
if opt.dataset == 'mnist' then ny = 10; imgExtension = '.png'
elseif opt.dataset == 'celebA' then ny = nil; imgExtension = '.jpg'; error('Not implemented.') end

-- Load nets
local generator = torch.load(opt.decNet)
local encoder = torch.load(opt.encNet)

local imgSz = {generator.output:size()[2], generator.output:size()[3], generator.output:size()[4]}

local inputX = torch.Tensor(opt.nImages, imgSz[1], imgSz[2], imgSz[3])
local Z = torch.Tensor(opt.nImages, opt.nz, 1, 1)
local Y = torch.Tensor(opt.nImages, ny):fill(-1)

-- Load to GPU
if opt.gpu > 0 then
    Z = Z:cuda(); Y = Y:cuda()
    cudnn.convert(generator, cudnn)
    cudnn.convert(encoder, cudnn)
    generator:cuda(); encoder:cuda()
else
    generator:float(); encoder:float()
end

generator:evaluate()
encoder:evaluate()

-- Load / generate X
if opt.loadOption == 0 then
  -- Generate X randomly from random Z and Y and then encoded it
  Z:normal(0,1)
  sampleY(Y, opt.dataset, ny, opt.nImages)
  inputX = generator:forward{Z, Y}:clone()
else
  -- Encode Z and Y from a given set of images
  inputX = obtainImageSet(inputX, opt.loadPath, opt.loadOption, imgExtension)
  if opt.gpu > 0 then inputX = inputX:cuda() end
end

local encOutput = encoder:forward(inputX)
Z = encOutput[1]; Y = encOutput[2]
Z:resize(Z:size(1), Z:size(2), 1, 1)
inputX = inputX:float() -- No longer needed in GPU

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
local sampleInput = {Z[{{1}}], Y[{{1}}]}
optnet.optimizeMemory(generator, sampleInput)

-- At this point, we have Z and Y and we need to expand them.
-- We just need to fix Z on rows and Y on columns
-- These ones are expanded version of Z and Y.
-- They just have more repetitions.
local nOutSamples = opt.nImages*ny
local outZ = torch.Tensor(nOutSamples, opt.nz, 1, 1)
local outY = torch.zeros(nOutSamples, ny)

if opt.gpu > 0 then outZ = outZ:cuda(); outY = outY:cuda() end
-- Fix Z for every row in generated samples.
-- A row has ny samples. Every i to (i-1)+ny samples outZ will have the same Z.
local j = 1
for i=1,nOutSamples,ny do
    outZ[{{i,(i-1)+ny},{},{},{}}] = Z[{{j},{},{},{}}]:expand(ny,opt.nz,1,1)
    j = j + 1
end

-- Fix Y for every column in generated samples.
sampleY(outY, opt.dataset, ny, nOutSamples)

-- Final image: 1st columns: original image (inputX)
--              2nd: reconstructed image (reconstX)
--              3rd-end: variations on Y (and same Z for each row) (outX)
local reconstX = generator:forward{Z, Y}:clone():float()
local outX = generator:forward{outZ, outY}:float()

local outputImage = torch.cat(inputX[{{1}}],reconstX[{{1}}], 1):cat(outX[{{1,ny}}],1)
for i=2,opt.nImages do
  local tmp = torch.cat(inputX[{{i}}],reconstX[{{i}}], 1):cat(outX[{{(i-1)*ny+1,i*ny}}],1)
  outputImage = outputImage:cat(tmp, 1)
end

disp.image(image.toDisplayTensor(outputImage,0,ny+2))
image.save(opt.name .. '.png', image.toDisplayTensor(outputImage,0,ny+2))


