--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

trainLoader = {}
local ySize = 10

-------- COMMON
local mnist = require 'mnist'
local trainSet = mnist.traindataset()
local labelDistr = torch.zeros(ySize)
for i=1,trainSet.size do
    labelDistr[trainSet.label[i]+1] = labelDistr[trainSet.label[i]+1] + 1
end
labelDistr:div(labelDistr:sum()) -- normalize to [0, 1]
--------------------------------------------------------------------------------------------
local loadSize   = {1, opt.loadSize}
local sampleSize = {1, opt.fineSize}

local function processImage(input)
   -- convert to float
   input = input:float()
   -- add 3rd dimension
   input:resize(1, input:size(1), input:size(2))
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(im)
   collectgarbage()
   local input = processImage(im)
   
   -- No data augmentation is performed on MNIST.
   -- This includes random crops and flips.
   input = image.scale(input, sampleSize[2], sampleSize[2])
   input:div(255):mul(2):add(-1) -- change [0, 255] to [-1, 1]
   return input
end

--------------------------------------
-- trainLoader
function trainLoader:sample(quantity)
    assert(quantity)
    local samples = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real images
    local labelsReal = torch.zeros(quantity, ySize) -- real label
    local labelsFake = torch.zeros(quantity, ySize) -- mismatch label (taken pseudo-randomly)
    
    -- Sampling with replacement (between batches we don't control which samples have been sampled)
    local randIdx = torch.randperm(trainSet.size):narrow(1,1,quantity)
    for i=1,quantity do
        -- Load and process image
        samples[{{i},{},{},{}}] = trainHook(trainSet.data[randIdx[i]])
        
        -- Compute real label
        local class = trainSet.label[randIdx[i]] -- MNIST label (0--9)
        labelsReal[{{i},{class+1}}] = 1 -- one-hot vector
        
        -- Compute randomly fake class. It can be any classe except the real one.
        local fakeClass = torch.randperm(ySize)
        if fakeClass[1] == class+1 then 
            fakeClass = fakeClass[2]
        else
            fakeClass = fakeClass[1]
        end
        labelsFake[{{i},{fakeClass}}] = 1
    end
    collectgarbage()
    
    return samples, labelsReal, labelsFake
end

function trainLoader:sampleY(quantity)
    -- TODO: MNIST interpolation?
    local y = torch.zeros(quantity, ySize)
    for i=1,quantity do
        local class = torch.multinomial(labelDistr,1)
        y[{{i},{class}}] = 1
    end
    collectgarbage()
    return y
end

function trainLoader:size()
    return trainSet.size
end

function trainLoader:ySize()
    return ySize
end

