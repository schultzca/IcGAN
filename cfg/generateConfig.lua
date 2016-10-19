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
    nz = 100,                 -- Z latent vector length 
    -- Conditional GAN parameters
    dataset = 'celebA'        -- dataset specification: mnist | celebA. It is necessary to know how to sample Y.          
}

elseif option == 1 then
-- Reconstruct images with encoder + generator and obtain variations on them (generation/reconstructWithVariations.lua)
opt = {
    nImages = 50,             -- number of samples to produce (only valid if loadOption != 1)
    decNet = 'checkpoints/',  -- path to the generator network
    encZnet = 'checkpoints/', -- path to encoder Z network
    encYnet = 'checkpoints/', -- path to encoder Y network
    gpu = 1,                  -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,                 -- Z latent vector length 
    loadOption = 2,           -- 0 = only generated images used, 1 = load input image, 2 = load multiple input images
    loadPath = '',            -- loadOption == 1: path to single image, loadOption==2: path to folder with images
    name = 'encoder_disentangle', -- name of output image
    -- Conditional GAN parameters
    dataset = 'celebA',       -- dataset specification: mnist | celebA. It is necessary to know how to sample Y. 
    threshold = true,         -- (celebA only) true= threshold original encoded Y to binary 
}


elseif option == 2 then
-- Attribute transfer: given two images, swap their attribute information Y (generation/attributeTransfer.lua)

elseif option == 3 then
-- Interpolate between two input images (generation/interpolate.lua)

else

  error("Option not recognized.") 

end