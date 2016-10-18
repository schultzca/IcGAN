require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

-- Load parameters from testConfig file
  assert(loadfile("cfg/testConfig.lua"))(0)
--torch.manualSeed(123)

local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

local ny -- Y label length. This depends on the dataset.
if opt.dataset == 'mnist' then ny = 10
elseif opt.dataset == 'celebA' then ny = 18 end

-- Load nets
local dec = torch.load(opt.decNet)
local encZ = torch.load(opt.encZnet)
local encY = torch.load(opt.encYnet)

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
    cudnn.convert(dec, cudnn)
    dec:cuda()
else
   dec:float()
end
dec:evaluate()
encZ:evaluate()
encY:evaluate()

local sampleInput = {inputZ:narrow(1,1,2), inputY:narrow(1,1,2)}
-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(dec, sampleInput)

-- Clone is needed, otherwise next forward call will overwrite inputX
local inputX = dec:forward{inputZ, inputY}:clone()
print('Images size: ', inputX:size(1)..' x '..inputX:size(2) ..' x '..inputX:size(3)..' x '..inputX:size(4))
--[[ Image to noise (encoder GAN) ]]--
-- Output noise should be equal to input noise

if opt.gpu > 0 then
    encZ:cuda()
    inputX = inputX:cuda()
    encY:cuda()
else
    encZ:float()
    encY:float()
end

local outY = encY:forward(inputX)
local outZ = encZ:forward(inputX)

print("Are input and output Z equal? ", torch.all(inputZ:eq(outZ)))
print('\tInput Z:  Mean, Stdv, Min, Max', inputZ:mean(), inputZ:std(), inputZ:min(), inputZ:max())
print('\tOutput Z: Mean, Stdv, Min, Max', outZ:mean(), outZ:std(), outZ:min(), outZ:max())
local error = torch.sum(torch.pow(torch.abs(inputZ-outZ),2))/(inputZ:size(1)*inputZ:size(2)*inputZ:size(3)*inputZ:size(4))
print('\tMSE per position Z: ', error)

print("Are input and output Y equal? ", torch.all(inputY:eq(outY)))
print('\tInput Y:  Mean, Stdv, Min, Max', inputY:mean(), inputY:std(), inputY:min(), inputY:max())
print('\tOutput Y: Mean, Stdv, Min, Max', outY:mean(), outY:std(), outY:min(), outY:max())
local error = torch.sum(torch.pow(torch.abs(inputY-outY),2))/(inputY:size(1)*inputY:size(2))
print('\tMSE per position Y: ', error)

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
    
    -- Encode image to attribute vector Y
    local outY = encY:forward(inputX)
    
    -- Encode image to latent representation Z
    outZ = encZ:forward(inputX)
    
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
              local val = outY[i][j]
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
local outX = dec:forward{outZ, outY}
local combinedX = torch.Tensor(opt.batchSize*2, outX:size(2), outX:size(3), outX:size(4))
local j = 1
for i=1,opt.batchSize*2,2 do
    combinedX[{{i}}]:copy(inputX[{{j}}])
    combinedX[{{i+1}}]:copy(outX[{{j}}])
    j = j + 1
end
-- Display input and output image
disp.image(image.toDisplayTensor(combinedX,0,2,true), {title='Reconstructed images'})

if opt.saveIm == true then
    image.save('inputImage.png', image.toDisplayTensor(inputX,0,torch.round(math.sqrt(opt.batchSize))))
    image.save('reconstructedImage.png', image.toDisplayTensor(outX,0,torch.round(math.sqrt(opt.batchSize))))
    image.save('encoder_reconstr_comparison.png', image.toDisplayTensor(combinedX,0,2,true))
end


