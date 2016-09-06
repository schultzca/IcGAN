require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
    im1Path = 'celebA/img_align_test/201971.jpg', -- path to image 1
    im2Path = 'celebA/img_align_test/201979.jpg', -- path to image 2
    decNet = 'checkpoints/c_celebA_64_filt_Yconv1_noTest_wrongYFixed_24_net_G.t7', --'checkpoints/c_celebA_64_filt_Yconv1_25_net_G.t7',--'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    encNet = 'checkpoints/encoder_c_celeba_Yconv1_noTest_7epochs.t7', --'checkpoints/encoder_c_celeba_Yconv1_noTanh_20epochs.t7',--'checkpoints/encoder128Filters2FC_dataset2_2_6epochs.t7',
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    nInterpolations = 4,
}

local function applyThreshold(Y, th)
    -- Takes a matrix Y and thresholds, given th, to -1 and 1
    assert(th>=-1 and th<=1, "Error: threshold must be between -1 and 1")
    for i=1,Y:size(1) do
        for j=1,Y:size(2) do
            local val = Y[{{i},{j}}][1][1]
            if val > th then
                Y[{{i},{j}}] = 1
            else
                Y[{{i},{j}}] = -1
            end
        end
    end
    
    return Y
end


if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

local ny = 18 -- Y label length. This depends on the dataset. 18 for CelebA

-- Load nets
local generator = torch.load(opt.decNet)
local encoder = torch.load(opt.encNet)

local imgSz = {generator.output:size()[2], generator.output:size()[3], generator.output:size()[4]}

local inputX = torch.Tensor(2, imgSz[1], imgSz[2], imgSz[3])
local Z = torch.Tensor(opt.nInterpolations+2, opt.nz, 1, 1)
local Y = torch.Tensor(opt.nInterpolations+2, ny):fill(-1)

inputX[{{1}}] = image.load(opt.im1Path)
inputX[{{2}}] = image.load(opt.im2Path)

inputX:mul(2):add(-1) -- change [0, 1] to [-1, 1]

-- Load to GPU
if opt.gpu > 0 then
    inputX = inputX:cuda(); Z = Z:cuda(); Y = Y:cuda()
    cudnn.convert(generator, cudnn)
    cudnn.convert(encoder, cudnn)
    generator:cuda(); encoder:cuda()
else
    generator:float(); encoder:float()
end

generator:evaluate()
encoder:evaluate()

-- Encode real images to Z and Y
local encOutput = encoder:forward(inputX)
local tmpZ = encOutput[1]; local tmpY = encOutput[2]

applyThreshold(tmpY,0)

tmpZ:resize(tmpZ:size(1), tmpZ:size(2), 1, 1)
local im1Z = tmpZ[{{1}}]; local im2Z = tmpZ[{{2}}]
local im1Y = tmpY[{{1}}]; local im2Y = tmpY[{{2}}]

-- Interpolate Z and Y
-- do a linear interpolation in Z and Y space between point A and point B
local weight  = torch.linspace(1, 0, opt.nInterpolations+2)
for i = 1, opt.nInterpolations+2 do
    Z:select(1, i):copy(im1Z * weight[i] + im2Z * (1 - weight[i]))
    Y:select(1, i):copy(im1Y * weight[i] + im2Y * (1 - weight[i]))
end

-- Generate interpolations
local outX = generator:forward{Z, Y}:float()

local container = torch.Tensor(opt.nInterpolations+4, imgSz[1], imgSz[2], imgSz[3])
container[{{1}}]:copy(inputX[{{1}}])
container[{{container:size(1)}}]:copy(inputX[{{2}}])
for i=1,opt.nInterpolations+2 do
    container[{{i+1}}]:copy(outX[{{i}}])
end

disp.image(image.toDisplayTensor(container,0,container:size(1)))
image.save('interpolations.png', image.toDisplayTensor(container,0,container:size(1)))
