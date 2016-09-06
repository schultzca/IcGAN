require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
    im1Path = 'celebA/img_align_test/202391.jpg',
    im2Path = 'celebA/img_align_test/202359.jpg',
    decNet = 'checkpoints/c_celebA_64_filt_Yconv1_noTest_wrongYFixed_24_net_G.t7', --'checkpoints/c_celebA_64_filt_Yconv1_25_net_G.t7',--'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    encNet = 'checkpoints/encoder_c_celeba_Yconv1_noTest_7epochs.t7', --'checkpoints/encoder_c_celeba_Yconv1_noTanh_20epochs.t7',--'checkpoints/encoder128Filters2FC_dataset2_2_6epochs.t7',
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
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
local Z = torch.Tensor(4, opt.nz, 1, 1)
local Y = torch.Tensor(4, ny):fill(-1)

inputX[{{1},{},{},{}}] = image.load(opt.im1Path)
inputX[{{2},{},{},{}}] = image.load(opt.im2Path)

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

local encOutput = encoder:forward(inputX)
local tmpZ = encOutput[1]; local tmpY = encOutput[2]
tmpZ:resize(tmpZ:size(1), tmpZ:size(2), 1, 1)

Z[{{1,2},{},{},{}}]:copy(tmpZ); Y[{{1,2},{}}]:copy(tmpY)

-- Switch Z and Y
Z[{{3,4},{},{},{}}]:copy(tmpZ)
Y[{{3},{}}]:copy(Y[{{2},{}}]); Y[{{4},{}}]:copy(Y[{{1},{}}])
applyThreshold(Y,0)
local outX = generator:forward{Z, Y}:float()

local container = torch.Tensor(6, imgSz[1], imgSz[2], imgSz[3])
container[{{1}}]:copy(inputX[{{1}}])
container[{{2}}]:copy(outX[{{1}}])
container[{{3}}]:copy(outX[{{3}}])
container[{{4}}]:copy(inputX[{{2}}])
container[{{5}}]:copy(outX[{{2}}])
container[{{6}}]:copy(outX[{{4}}])
disp.image(image.toDisplayTensor(container,0,3))
image.save('attributeTransfer.png', image.toDisplayTensor(container,0,3))
print('Done!')
