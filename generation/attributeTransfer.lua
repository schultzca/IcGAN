-- Attribute transfer: given two images, swap their attribute information Y
-- Output image:
--  1st column: real image
--  2nd column: reconstructed image
--  3rd column: reconstructed image with swapped Y

require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

-- Load parameters from config file
assert(loadfile("cfg/generateConfig.lua"))(2)
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
local Z = torch.Tensor(4, opt.nz, 1, 1)
local Y = torch.Tensor(4, ny):fill(-1)

inputX[{{1},{},{},{}}] = image.load(opt.im1Path)
inputX[{{2},{},{},{}}] = image.load(opt.im2Path)

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

local tmpZ = encZ:forward(inputX)
local tmpY = encY:forward(inputX)
tmpZ:resize(tmpZ:size(1), tmpZ:size(2), 1, 1)

Z[{{1,2},{},{},{}}]:copy(tmpZ); Y[{{1,2},{}}]:copy(tmpY)

-- Switch Z and Y
Z[{{3,4},{},{},{}}]:copy(tmpZ)
Y[{{3},{}}]:copy(Y[{{2},{}}]); Y[{{4},{}}]:copy(Y[{{1},{}}])
applyThreshold(Y,0)
local outX = generator:forward{Z, Y}:float()

local container = torch.Tensor(6, opt.loadSize[1], opt.loadSize[2], opt.loadSize[3])
container[{{1}}]:copy(inputX[{{1}}])
container[{{2}}]:copy(outX[{{1}}])
container[{{3}}]:copy(outX[{{3}}])
container[{{4}}]:copy(inputX[{{2}}])
container[{{5}}]:copy(outX[{{2}}])
container[{{6}}]:copy(outX[{{4}}])
disp.image(image.toDisplayTensor(container,0,3))
image.save('attributeTransfer.png', image.toDisplayTensor(container,0,3))
print('Done!')
