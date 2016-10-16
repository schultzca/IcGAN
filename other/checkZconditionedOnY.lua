-- Check whether Z encoded vectors from real images is dependant on Y. 
-- That is, check if the distribution of Z changes with different Ys.
-- This codes requires to the file "isZconditionedOnY.dmp" (obtained from a modified
-- version of generateReconstructedDataset.lua), which needs to contain:
--  · Z = encoded vectors of real images
--  · Yreal = real labels Y from the real images (binary, -1: disabled, 1: enabled)
--  · Ygen = encoded labels Y' from the real images (optional) (non binary, ideally -1: disabled, 1: enabled)

-- Load file containing Z and real Y
local matio = require "matio"
local data = torch.load('celebA/isZconditionedOnY.dmp')
local Z     = data.Z
local Yreal = data.Yreal
local Ygen  = data.Ygen
local Zmean = torch.mean(Z,1)
for k=1,Z:size(2) do
    Z[{{},{k}}]:add(-Zmean[1][k])
end
Z = Z * 1.8162
for k=1,Z:size(2) do
    Z[{{},{k}}]:add(Zmean[1][k])
end
--matio.save('Z.mat',Z)
--torch.save('celebA/Zmean.dmp', torch.mean(Z,1))
local sz = Z:size(1)

-- Z1: Zs where attribute blonde is activated
-- Create mask of positions where blonde is 1
local attIdx = 4 -- Blond hair
-- Torch doesn't support boolean indexing. You need to transform
-- a binary mask into the indices of the positions you want to keep
local idx = torch.linspace(1,sz,sz):long()
local mask = idx[Yreal[{{},{attIdx}}]:gt(0)]
local Z1 = Z:index(1,mask)

-- Z2: Zs where attribute dark hair is activated
attIdx = 3-- Black hair
mask = idx[Yreal[{{},{attIdx}}]:gt(0)]
local Z2 = Z:index(1,mask)

print(("Differences of std and mean on the %d dimensions of Z1 (%d elements) and Z2 (%d elements):"):format(Z1:size(2), Z1:size(1),Z2:size(1)))
local Z1std = Z1:std(1):resize(Z1:size(2))
local Z2std = Z2:std(1):resize(Z2:size(2))
local dif = (Z1std-Z2std):abs()
print(("Std error:  Mean, max, min: %.4f, %.4f, %.4f"):format(dif:mean(), dif:max(), dif:min()))
local Z1m = Z1:mean(1):resize(Z1:size(2))
local Z2m = Z2:mean(1):resize(Z2:size(2))
local dif = (Z1m-Z2m):abs()
print(("Mean error: Mean, max, min: %.4f, %.4f, %.4f"):format(dif:mean(), dif:max(), dif:min()))
-- Maximum difference: 0.3477 (mean) and 0.065 (std)
print(Z1m:mean(), Z1std:mean())

print(Z2m:mean(), Z2std:mean())

print("\nMean of whole Z: ")
print(("\t %.4f +- %.4f"):format(Z:mean(), Z:std()))
print("Mean per dimension of whole Z: ")
print(("\t %.4f +- %.4f"):format(Z:mean(1):mean(), Z:mean(1):std()))
print("Std per dimension of whole Z: ")
print(("\t %.4f +- %.4f"):format(Z:std(1):mean(), Z:std(1):std()))

-- This is used to retrain the encoder with this new distribution of Zs
--local data = {}
--data.Zmean = Z:mean(1):resize(Z:size(2))
--data.Zstd = Z:std(1):resize(Z:size(2))
--torch.save('celebA/encoded_Z_distribution_real_images.dmp', data)
