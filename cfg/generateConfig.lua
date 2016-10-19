-- ==CONFIGURATION FILE FOR IMAGE GENERATION==
local option = ... -- the argument is received as varargs, hence the ...

assert(option, 'Option not specified.')

if option == 0 then
-- Generate images from scratch using the generator only, no encoder involved (generation/generate.lua)
opt = {
    batchSize = 1000,         -- number of samples to produce (it should be multiple of Y attribute vector length (ny))
    noisetype = 'normal',     -- type of noise distribution (uniform / normal).
    net = 'checkpoints/',     -- path to the generator network
    imsize = 1,               -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',     -- random / line / linefull1d / linefull
    name = 'gen_samples',     -- name of the file saved
    gpu = 1,                  -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,              -- Display image: 0 = false, 1 = true
    nz = 100,              
    -- Conditioned GAN parameters
    dataset = 'celebA'        -- mnist | celebA           
}

elseif option == 1 then
-- Reconstruct images with encoder + generator and obtain variations on them (generation/reconstructWithVariations.lua)

elseif option == 2 then
-- Attribute transfer: given two images, swap their attribute information Y (generation/attributeTransfer.lua)

elseif option == 3 then
-- Interpolate between two input images (generation/interpolate.lua)

else

  error("Option not recognized.") 

end