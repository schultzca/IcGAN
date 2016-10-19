-- ==CONFIGURATION FILE FOR TRAINING==
local option = ... -- the argument is received as varargs, hence the ...

assert(option, 'Option not specified.')

if option == 0 then
-- GAN parameters (trainGAN.lua)
opt = {
   name = 'celeba',        -- name used to store the discriminator and generator
   dataset = 'celebA',     -- celebA | mnist
   batchSize = 64,
   loadSize = 64,          -- images will be scaled up/down to this size
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in last deconv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   niter = 25,             -- #  of epochs
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   noise = 'normal',       -- uniform / normal
   dataRoot = 'celebA',    -- path to the dataset images, except for not mnist. If mnist, just put 'mnist'
   randomCrop = false,     -- true = crop randomly the samples of the dataset (celebA only)
   fineSize = 64,          -- size of the image crop (as long as randomCrop == true)
   -- Miscellaneous parameters
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   saveGif = 0,            -- saveGif = 1 saves images of the generated samples progress to later create a gif
   poweroff = 0,           -- 1 = power off computer after training, 0 = not power off
   -- Parameters for conditioned GAN
   trainWrongY = true      -- explicitly train discriminator with real images and wrong Y (mismatching Y)
}

elseif option == 1 then
-- Encoder Z parameters (trainEncoder.lua)
opt = {
  name = 'encoderZ_celeba',    -- name of the encoder
  type = 'Z',                  -- encoder type. 'Z': encoder Z: image X to latent representation Z | 'Y': encoder Y: image X to attribute vector Y
  batchSize = 64,
  outputPath = 'checkpoints/', -- path used to store the encoder network
  datasetPath = 'celebA/genDataset/', -- folder where the dataset files are stored (not the files themselves)
                              -- for encoder Z you need the file grountruth.dmp (obtained with data/generateEncoderDataset.lua)
  split = 0.66,               -- split between train and test (e.g. 0.66 = 66% train, 33% test)
  nConvLayers = 4,            -- # of convolutional layers on the net
  nf = 32,                    -- # of filters in hidden layer
  FCsz = nil,                 -- size of the last fully connected layer. If nil, size will be the same as previous FC layer.
  nEpochs = 15,               -- # of epochs
  lr = 0.0001,                -- initial learning rate for adam
  beta1 = 0.1,                -- momentum term of adam
  gpu = 1,                     -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  -- Miscellaneous
  display = 1,                -- display 1 = train and test error, 2 = error + batch input images, 0 = false
  poweroff = 0,               -- 1 = power off computer after training, 0 = not power off
        
  }
elseif option == 2 then
-- Encoder Y parameters (trainEncoder.lua)
opt = {
  name = 'encoderY_c_celeba',  -- name of the encoder
  type = 'Y',                  -- encoder type. 'Z': encoder Z: image X to latent representation Z | 'Y': encoder Y: image X to attribute vector Y
  batchSize = 64,
  outputPath = 'checkpoints/', -- path used to store the encoder network
  datasetPath = 'celebA/genDataset/', -- folder where the dataset files are stored (not the files themselves)
                              -- for encoder Y you need the file images.dmp (data/preprocess_celebA.lua) and imLabels.dmp (data/donkey_celebA.lua)
  split = 0.66,               -- split between train and test (e.g. 0.66 = 66% train, 33% test)
  nConvLayers = 4,            -- # of convolutional layers on the net
  nf = 32,                    -- # of filters in hidden layer
  FCsz = 512,                 -- size of the last fully connected layer. If nil, size will be the same as previous FC layer.
  nEpochs = 15,               -- # of epochs
  lr = 0.0001,                -- initial learning rate for adam
  beta1 = 0.1,                -- momentum term of adam
  gpu = 1,                    -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  -- Miscellaneous
  display = 1,                -- display 1 = train and test error, 2 = error + batch input images, 0 = false
  poweroff = 0,               -- 1 = power off computer after training, 0 = not power off  
  
  }
elseif option == 3 then
-- Generate encoder dataset parameters necessary to train the encoder Z (generateEncoderDataset.lua)
opt = {
    samples = 202599-19961,  -- total number of samples to generate
    batchSize = 256,         -- number of samples to produce at the same time
    noisetype = 'normal',    -- type of noise distribution (uniform / normal).
    net = 'checkpoints/',    -- path to the generator network
    imsize = 1,              -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px,
    gpu = 1,                 -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,                -- size of Z vector
    outputFolder = 'celebA/genDataset/', -- path where the dataset will be stored
    outputFormat = 'binary', -- dataset output format (binary | ascii). Binary is faster, but platform-dependent.
    storeAsTensor = true,    -- true = store images as tensor (recommended), false = store images as images (lossy)
    -- Conditional GAN parameters
    dataset = 'celebA',      -- mnist | celebA  
  }

else

  error("Option not recognized.") 

end