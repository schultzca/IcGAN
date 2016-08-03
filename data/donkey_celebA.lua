--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'io'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.dataRoot'
if not paths.dirp(opt.dataRoot) then
    error('Did not find directory: ' .. opt.dataRoot)
end

trainLoader = {}
-- Index of the attributes from celebA that will be ignored
local attrFil = {1,2,3,4,7,8,11,14,15,20,24,25,26,28,30,31,35,37,38,39,40}
-- Load all image paths on opt.dataRoot folder
local imPaths = torch.load(opt.dataRoot..'/imNames.dmp')
print('Loading images from '..opt.dataRoot..'/images.dmp')
local images = torch.load(opt.dataRoot..'/images.dmp')
print(('Done. Loaded %.2f GB.'):format((4*images:size(1)*images:size(2)*images:size(3)*images:size(4))/2^30))

-- Load each label vector for each image
local imLabels
local labelPath = opt.dataRoot..'/list_attr_celeba.txt'
local file = io.open(labelPath, 'r')

file:read() -- Skip 1st line
local rawLabelHeader = file:read() -- 2nd line is header
local labelHeader = {}
-- Read header
local i = 1
local j = 1
for label in rawLabelHeader:gmatch('%S+') do -- split on space
    if i ~= attrFil[j] then
        labelHeader[#labelHeader+1] = label
    else
        j = j + 1
    end
    i = i + 1
end

imLabels = torch.IntTensor(#imPaths, #labelHeader)

local ySize = #labelHeader

-- Read the rest of the file
local i = 1
local skippedImages = 0 -- Controls if some images are missing
for line in file:lines() do  
  local l = line:split('%s+')
  -- Check if image in labels file exists on imPaths
  -- If not, skip it from imLabels
  local imName = imPaths[i-skippedImages]
  if imName == l[1] then
      local j = 2 -- indexs line l. First element of line is imName, we skip it
      local k = 1 -- index attrFil. Just increments when an filtered attribute is found
      local k2 = 1 -- indexs imLabels with filtered labels
      while k2 <= #labelHeader do
          if j-1 ~= attrFil[k] then
              local val = l[j]
              imLabels[{{i-skippedImages},{k2}}] = tonumber(val)
              k2 = k2 + 1
          else
              k = k + 1
          end
          j = j + 1
      end
  else
      print("Warning: "..l[1].." appears on labels file but hasn't been found in "..opt.dataRoot)
      skippedImages = skippedImages + 1
  end
  i = i + 1
end

-- Narrow imLabels tensor in case some images are missing
if skippedImages > 0 then imLabels = imLabels:narrow(1,1,imLabels:size(1)-skippedImages) end

file:close()

-- Calculate probabilities from each attribute
local attrProb = torch.sum(imLabels:clone():add(1):div(2), 1):float():div(imLabels:size(1))
attrProb:resize(attrProb:size(2))
--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(input)
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

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(input)
   --collectgarbage()
   local input = loadImage(input)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   
   local out
   -- Random crop is optional.
   if opt.randomCrop then
     -- do random crop
     local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
     local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
     out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   else
     out = image.scale(input, oW, oH)
   end
   
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end

--------------------------------------
-- trainLoader
function trainLoader:sample(quantity)
    assert(quantity)
    assert(quantity*2<=#imPaths, ("Batch size can't be greater than %d"):format(#imPaths/2))
    local samples = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real images
    local labelsReal = torch.zeros(quantity, ySize) -- real label
    local labelsFake = torch.zeros(quantity, ySize) -- mismatch label (taken pseudo-randomly)
    
    -- Sampling with replacement. Between batches we don't control which samples have been sampled
    local randIdx = torch.randperm(#imPaths):narrow(1,1,quantity*2)
    for i=1,quantity do
        -- Load and process image
        samples[{{i},{},{},{}}] = trainHook(images[randIdx[i]])
        
        -- Compute real label
        labelsReal[{{i},{}}]  = imLabels[{{randIdx[i]},{}}]
        
        -- Compute randomly fake class. It can be any classe except the real one.
        labelsFake[{{i},{}}] = imLabels[{{randIdx[quantity+i]},{}}]
        if torch.all(labelsReal[{{i},{}}]:eq(labelsFake[{{i},{}}])) then 
        -- If labelsFake happen to be equal to labelsReals, 
        -- alter randomly some of labelsFake positions (-1 to 1 or 1 to -1)
            local randPosition = math.random(ySize)
            labelsFake[{{i},{randPosition}}] = -labelsFake[{{i},{randPosition}}]
        end
    end
    collectgarbage()
    
    return samples, labelsReal, labelsFake
end

function trainLoader:sampleY(quantity)
    local y = torch.zeros(quantity, ySize)
    
    -- Get real labels
    local randIdx = torch.randperm(#imPaths):narrow(1,1,quantity)
    y = imLabels:index(1, randIdx:long())
    
    collectgarbage()
    return y
end

function trainLoader:size()
    return #imPaths
end

function trainLoader:ySize()
    return ySize
end
