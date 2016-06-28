require 'image'
require 'nn'
disp = require 'display'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,         -- number of samples to produce
    decNet = 'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    encNet = 'checkpoints/encoder128Filters2FC_dataset2_6epochs.t7',
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px,
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    customInputImage = 2,  -- 0 = no custom, only generated images used, 1 = load input image, 2 = load multiple input images
    customImagesPath = 'mnist/images', -- path used wehn customInputImage is 1 (path to single image) or 2 (path to folder with images)
}
torch.manualSeed(123)

-- Load net
local decG = util.load(opt.decNet, opt.gpu)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(decG:get(1)) == 'nn.View' then
    decG:remove(1)
end

local encG = util.load(opt.encNet, opt.gpu)

--[[ Noise to image (decoder GAN) ]]--
local inNoise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
inNoise:normal(0, 1)

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    decG:cuda()
    util.cudnn(decG)
    inNoise = inNoise:cuda()
else
   decG:float()
end
decG:evaluate()
encG:evaluate()

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
util.optimizeInferenceMemory(decG)

-- Clone is needed, otherwise next forward call will overwrite inImage
local inImage = decG:forward(inNoise):clone()

--[[ Image to noise (encoder GAN) ]]--
-- Output noise should be equal to input noise

if opt.gpu > 0 then
    encG:cuda()
    inImage = inImage:cuda()
else
    encG:float()
end

local outNoise = encG:forward(inImage)

print("Are input and output noise equal? ", torch.all(inNoise:eq(outNoise)))
print('\tinNoise: Mean, Stdv, Min, Max', inNoise:mean(), inNoise:std(), inNoise:min(), inNoise:max())
print('\toutNoise: Mean, Stdv, Min, Max', outNoise:mean(), outNoise:std(), outNoise:min(), outNoise:max())
local error = torch.sum(torch.abs(inNoise-outNoise))/opt.nz
print('\tAbsolute error per position: ', error)

-- Now test if an encoded and then decoded image looks similar to the input image
if opt.customInputImage > 0 then
    if opt.customInputImage == 1 then
        -- Load input image X (optional)
        local tmp = image.load(opt.customImagesPath)
        tmp = image.scale(tmp, inImage:size(3), inImage:size(4))
        -- Image dimensions is 3. We need a 4th dimension indicating the number of images.
        tmp:resize(1, inImage:size(2), inImage:size(3), inImage:size(4))
        inImage = tmp
    else
        -- Load multiple images given a path
        assert(paths.dir(opt.customImagesPath)~=nil, "customImagesPath is not a directory")
        local i = 1
        local fileIterator = paths.files(opt.customImagesPath, '.png')
        while i <= opt.batchSize do
            local imPath = opt.customImagesPath .. '/' .. fileIterator()
            local im = image.load(imPath)
            inImage[{{i},{},{},{}}] = image.scale(im, inImage:size(3), inImage:size(4))
            i = i + 1
        end
    end

    print('Images size: ', inImage:size(1)..' x '..inImage:size(2) ..' x '..inImage:size(3)..' x '..inImage:size(4))
    
    if opt.gpu > 0 then
      inImage = inImage:cuda()
    end
    
    -- Encode it to noise Z
    outNoise = encG:forward(inImage)
end

outNoise:resize(outNoise:size(1), outNoise:size(2), 1, 1)

-- Decode it to an output image X2
local outImage = decG:forward(outNoise)

-- Display input and output image
disp.image(inImage, {title='Input image'})
disp.image(outImage, {title='Encoded and decoded image'})
print("Are input and output images equal? ", torch.all(inImage:eq(outImage)))
image.save('inputImage.png', image.toDisplayTensor(inImage,0,torch.round(math.sqrt(opt.batchSize))))
image.save('reconstructedImage.png', image.toDisplayTensor(outImage,0,torch.round(math.sqrt(opt.batchSize))))

