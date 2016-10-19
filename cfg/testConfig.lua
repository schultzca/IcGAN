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
-- Evaluate model parameters (evaluation/evaluateModel.lua)
opt = {
    batchSize = 64,
    net = 'checkpoints/',                               -- path to the Attribute Predictor network (encoder Y: given an image X, outputs the conditional info Y).
    gpu = 1,                                            -- gpu mode. 0 = CPU, 1 = GPU
    testSetPath = 'celebA/recDataset/groundtruth.dmp',  -- path to file containing reconstructed images X to and the conditional info Y of the *original* images
}

elseif option == 2 then
-- Generate reconstructed dataset (generateReconstructedDataset.lua) parameters to evaluate the whole model with evaluateModel.lua
opt = {
      batchSize = 18,
      decNet = 'checkpoints/',  -- path to generator network
      encZnet = 'checkpoints/', -- path to encoder Z network
      encYnet = 'checkpoints/', -- path to encoder Y network
      gpu = 1,                  -- gpu mode. 0 = CPU, 1 = GPU
      display = 0,              -- 1 = display output reconstructed images, 0 = not display
      nz = 100,                 -- size of Z vector
      path = 'celebA/',         -- path to folder where a file named im_and_labels_test_set.dmp contains *test* images and their attribute vectors Y.
      outputFolder = 'celebA/recDataset/', -- path where the dataset will be stored
      threshold = true,         -- threshold Y vectors to binary or not
  }

elseif option == 3 then

else

  error("Option not recognized.") 

end