require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

assert(loadfile("cfg/generateConfig.lua"))(3)
-- one-line argument parser. Parses environment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

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
local encZ = torch.load(opt.encZnet)
local encY = torch.load(opt.encYnet)

local inputX = torch.Tensor(2, opt.loadSize[1], opt.loadSize[2], opt.loadSize[3])
local Z = torch.Tensor(opt.nInterpolations+2, opt.nz, 1, 1)
local Y = torch.Tensor(opt.nInterpolations+2, ny):fill(-1)

inputX[{{1}}] = image.load(opt.im1Path)
inputX[{{2}}] = image.load(opt.im2Path)

inputX:mul(2):add(-1) -- change [0, 1] to [-1, 1]

-- Load to GPU
if opt.gpu > 0 then
    inputX = inputX:cuda(); Z = Z:cuda(); Y = Y:cuda()
    cudnn.convert(generator, cudnn)
    cudnn.convert(encZ, cudnn); cudnn.convert(encY, cudnn)
    generator:cuda(); encZ:cuda(); encY:cuda()
else
    generator:float(); encZ:float(); encY:float()
end

generator:evaluate()
encZ:evaluate(); encY:evaluate()

-- Encode real images to Z and Y
local tmpZ = encZ:forward(inputX)
local tmpY = encY:forward(inputX)

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

local container = torch.Tensor(opt.nInterpolations+4, opt.loadSize[1], opt.loadSize[2], opt.loadSize[3])
container[{{1}}]:copy(inputX[{{1}}])
container[{{container:size(1)}}]:copy(inputX[{{2}}])
for i=1,opt.nInterpolations+2 do
    container[{{i+1}}]:copy(outX[{{i}}])
end

disp.image(image.toDisplayTensor(container,0,container:size(1)))
image.save('interpolations.png', image.toDisplayTensor(container,0,container:size(1)))
