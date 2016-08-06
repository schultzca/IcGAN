require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
    batchSize = 64,         -- number of samples to produce
    decNet = 'checkpoints/c_celebA_64_filt_Yconv1_25_net_G.t7',--'checkpoints/c_mnist_25_net_G.t7' 'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    encNet = 'checkpoints/encoder_c_celeba_Yconv1_noTanh_20epochs.t7',--'checkpoints/encoder_c_mnist_6epochs.t7' 'checkpoints/encoder128Filters2FC_dataset2_2_6epochs.t7',
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    customInputImage = 2,  -- 0 = no custom, only generated images used, 1 = load input image, 2 = load multiple input images
    customImagesPath = 'celebA/img_align_celeba/', --'mnist/images', -- path used when customInputImage is 1 (path to single image) or 2 (path to folder with images)
    -- Conditional GAN parameters
    dataset = 'celebA',
    threshold = true, -- threshold Y vectors to binary or not
}
torch.manualSeed(123)

local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

local ny -- Y label length. This depends on the dataset.
if opt.dataset == 'mnist' then ny = 10
elseif opt.dataset == 'celebA' then ny = 19 end

-- Load nets
local decG = torch.load(opt.decNet)
local encG = torch.load(opt.encNet)

--[[ Noise to image (decoder GAN) ]]--
local inputZ = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
local inputY = torch.Tensor(opt.batchSize, ny):fill(-1)

-- Y is specific for MNIST dataset
for i=1,opt.batchSize do
  inputY[{{i},{((i-1)%ny)+1}}] = 1
end

inputZ:normal(0, 1)

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    inputZ = inputZ:cuda(); inputY = inputY:cuda()
    cudnn.convert(decG, cudnn)
    decG:cuda()
else
   decG:float()
end
decG:evaluate()
encG:evaluate()

local sampleInput = {inputZ:narrow(1,1,2), inputY:narrow(1,1,2)}
-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(decG, sampleInput)

-- Clone is needed, otherwise next forward call will overwrite inputX
local inputX = decG:forward{inputZ, inputY}:clone()
print('Images size: ', inputX:size(1)..' x '..inputX:size(2) ..' x '..inputX:size(3)..' x '..inputX:size(4))
--[[ Image to noise (encoder GAN) ]]--
-- Output noise should be equal to input noise

if opt.gpu > 0 then
    encG:cuda()
    inputX = inputX:cuda()
else
    encG:float()
end

local output = encG:forward(inputX)
local outZ = output[1]; local outY = output[2]

print("Are input and output Z equal? ", torch.all(inputZ:eq(outZ)))
print('\tInput Z:  Mean, Stdv, Min, Max', inputZ:mean(), inputZ:std(), inputZ:min(), inputZ:max())
print('\tOutput Z: Mean, Stdv, Min, Max', outZ:mean(), outZ:std(), outZ:min(), outZ:max())
local error = torch.sum(torch.abs(inputZ-outZ))/(inputZ:size(1)*inputZ:size(2)*inputZ:size(3)*inputZ:size(4))
print('\tAbsolute error per position: ', error)

print("Are input and output Y equal? ", torch.all(inputY:eq(outY)))
print('\tInput Y:  Mean, Stdv, Min, Max', inputY:mean(), inputY:std(), inputY:min(), inputY:max())
print('\tOutput Y: Mean, Stdv, Min, Max', outY:mean(), outY:std(), outY:min(), outY:max())
local error = torch.sum(torch.abs(inputY-outY))/(inputY:size(1)*inputY:size(2))
print('\tAbsolute error per position: ', error)

-- Now test if an encoded and then decoded image looks similar to the input image
if opt.customInputImage > 0 then
    if opt.customInputImage == 1 then
        -- Load input image X (optional)
        local tmp = image.load(opt.customImagesPath)
        tmp = image.scale(tmp, inputX:size(3), inputX:size(4))
        -- Image dimensions is 3. We need a 4th dimension indicating the number of images.
        tmp:resize(1, inputX:size(2), inputX:size(3), inputX:size(4))
        inputX = tmp
    else
        -- Load multiple images given a path
        assert(paths.dir(opt.customImagesPath)~=nil, "customImagesPath is not a directory")
        local i = 1
        local filter = function(filename) 
                  for i=1, #extensionList do
                      if filename:find(i) then return true end
                  end
                  return false
              end
        local fileIterator = paths.files(opt.customImagesPath, filter)
        while i <= opt.batchSize do
            local imPath = opt.customImagesPath .. '/' .. fileIterator()
            local im = image.load(imPath)
            inputX[{{i},{},{},{}}] = image.scale(im, inputX:size(3), inputX:size(4))
            i = i + 1
        end
    end
    inputX:mul(2):add(-1) -- change [0, 1] to [-1, 1]
    print('Images size: ', inputX:size(1)..' x '..inputX:size(2) ..' x '..inputX:size(3)..' x '..inputX:size(4))

    if opt.gpu > 0 then
      inputX = inputX:cuda()
    end
    
    -- Encode it to noise Z
    output = encG:forward(inputX)
    local outZ = output[1]; local outY = output[2]
end

outZ:resize(outZ:size(1), outZ:size(2), 1, 1)

-- (Optional) Threshold Y
if opt.threshold then
  if string.lower(opt.dataset) == 'mnist' then
    -- Convert to one-hot vector
      local tmp = torch.Tensor(1,ny):fill(-1)
      for i=1,outY:size(1) do
          tmp:fill(-1)
          local _, maxIdx = torch.max(outY[{{i},{}}],2)
          tmp[{{},{maxIdx[1][1]}}] = 1
          outY[{{i},{}}] = tmp:clone()
      end
  else
      -- celebA
      for i=1,outY:size(1) do
          for j=1,outY:size(2) do
              local val = outY[{{i},{j}}][1][1]
              if val > 0 then
                  outY[{{i},{j}}] = 1
              else
                  outY[{{i},{j}}] = -1
              end
          end
      end
  end
end
-- Decode it to an output image X2
local outX = decG:forward{outZ, outY}

-- Display input and output image
disp.image(inputX, {title='Input image'})
disp.image(outX, {title='Encoded and decoded image'})
print("Are input and output images equal? ", torch.all(inputX:eq(outX)))
image.save('inputImage.png', image.toDisplayTensor(inputX,0,torch.round(math.sqrt(opt.batchSize))))
image.save('reconstructedImage.png', image.toDisplayTensor(outX,0,torch.round(math.sqrt(opt.batchSize))))

