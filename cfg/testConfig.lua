-- ==CONFIGURATION FILE FOR TESTING/EVALUATION==
local option = ... -- the argument is received as varargs, hence the ...

assert(option, 'Option not specified.')

if option == 0 then
-- Test encoder parameters (evaluation/testEncoder.lua)
opt = {
    batchSize = 500,          -- number of samples to produce
    decNet = 'checkpoints/',  -- path to generator network
    encZnet = 'checkpoints/', -- path to encoder Z network
    encYnet = 'checkpoints/', -- path to encoder Z network
    gpu = 1,                  -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,                 -- Z latent vector size
    customInputImage = 2,     -- 0 = no custom, only generated images used, 1 = load input image, 2 = load multiple input images
    customImagesPath = 'celebA/img_align_test/', -- path used when customInputImage is 1 (path to single image) or 2 (path to folder with images)
    saveIm = false,           -- true = save output images
    -- Conditional GAN parameters
    dataset = 'celebA',       -- celebA | mnist
    threshold = true,         -- threshold Y vectors to binary or not
}

elseif option == 1 then
-- Encoder Z parameters (trainEncoder.lua)

elseif option == 2 then
-- Generate encoder dataset parameters necessary to train the encoder Z (generateEncoderDataset.lua)

elseif option == 3 then

else

  error("Option not recognized.") 

end