-- ==CONFIGURATION FILE==
local option = ... -- the argument is received as varargs, hence the ...

assert(option, 'Option not specified.')

if option == 0 then
-- GAN parameters (trainGAN.lua)
opt = {
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
   name = 'celebA',        -- name used to store the discriminator and generator
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

else

  error("Option not recognized.") 

end