-- This file loads an already trained GAN and generate a dataset that
-- consists of generated images and the input noise Z that generated them.
-- This dataset will be later used for training an encoding GAN (given an image,
-- encode it as a noise vector Z so as the decoding GAN can generate the same image
-- from it).

require 'image'
require 'nn'
require "lfs"
util = paths.dofile('../util.lua')
torch.setdefaulttensortype('torch.FloatTensor')
--torch.manualSeed(123) -- This only works if gpu == 0

function getParameters()
  local opt = {
    samples = 10000,          -- total number of samples to generate
    batchSize = 256,         -- number of samples to produce at the same time
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = 'checkpoints/experiment1_10_net_G.t7',-- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px,
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,               -- size of noise vector
    outputFolder = 'mnist/generatedDataset/', -- path where the dataset will be stored
    outputFormat = 'binary' -- (binary | ascii) binary is faster, but platform-dependent.
  }
  
  for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
  
  assert(opt.samples >= opt.batchSize, "Batch size (opt.batchSize) can't be greater than number of samples (opt.samples).")
  
  return opt
end

function initializeNet(net, opt)
  -- Initializes the net and the input noise.
  -- This involves passing the network to GPU and other optimizations.
  
   -- for older models, there was nn.View on the top
  -- which is unnecessary, and hinders convolutional generations.
  if torch.type(net:get(1)) == 'nn.View' then
      net:remove(1)
  end
  
  local noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
  
  if opt.gpu > 0 then
      require 'cunn'
      require 'cudnn'
      net:cuda()
      util.cudnn(net)
      noise = noise:cuda()
  else
      net:float()
  end
  
  util.optimizeInferenceMemory(net)
  
  return net, noise
end

function createFolder(path)
    local state, error
    
    state, error = lfs.mkdir(path)
    assert(state or error == 'File exists',"Couldn't create directory "..path) 
end

function saveData2Txt(imageSet, noise, path, idx)
-- imageSet dimension: # of images x im3 x im2 x im1
-- noise: # of images x nz x 1 x 1
-- path: folder where the images will be stored
-- idx: index used to name the image file

    -- Open dataset file
    file = io.open(path..'gt.txt', 'a')
    
    for j=1,imageSet:size(1) do
        local outputFile = string.format(path.."%.7d.png",idx+j-1)
        local im = imageSet[{{j},{},{},{}}]
        -- Save image
        im:resize(im:size(2), im:size(3), im:size(4)) -- 4 to 3 dimensions
        image.save(outputFile, im)
        -- Write path to file
        file:write(outputFile, "\t")
        -- Write noise to file
        file:write(noise[{{j},{},{},{}}]:type('torch.CharTensor'), "\n")
    end

    file:close()
end

function saveData(imageSet, noise, outFile, path, idx)
-- imageSet dimension: # of images x im3 x im2 x im1
-- noise: # of images x nz x 1 x 1
-- path: folder where the images will be stored
-- idx: index used to name the image file

    for j=1,imageSet:size(1) do
        local i = idx+j-1
        local imageName = string.format("%.7d.png",i)
        local imagePath = path..'images/'..imageName
        local im = imageSet[{{j},{},{},{}}]
        -- Save image
        im:resize(im:size(2), im:size(3), im:size(4)) -- 4 to 3 dimensions
        image.save(imagePath, im)
        -- Update output file
        outFile.imNames[i] = imageName
        outFile.noises[{{i},{},{},{}}] = noise[{{j},{},{},{}}]:float() -- float() is necessary in case noise is in GPU
    end
end

function main()

  opt = getParameters()
  
  -- Load generative network
  local net = util.load(opt.net, opt.gpu)
  local noise
  
  net, noise = initializeNet(net, opt)
  
  -- Create output folder path and other subfolders
  createFolder(opt.outputFolder)
  createFolder(opt.outputFolder..'images/')
  
  -- Note: If opt.samples is not divisble by opt.batchSize,
  -- more samples than opt.samples will be produced.
  local nSamplesReal = math.floor(opt.samples/opt.batchSize)*opt.batchSize+opt.batchSize%opt.samples
  
  -- Initialize output file that will contain all the information
  local outFile = {
    relativePath = opt.outputFolder..'images/', -- write relative path once, not for every image
    imNames = {},
    -- Change to torch.Tensor later. Now torch.zeros for debbuging purposes
    noises = torch.Tensor(nSamplesReal, opt.nz, opt.imsize, opt.imsize)
  }
  
  -- Create as many pairs of generated image and noise vector
  -- as specified by opt.samples.
  for i=1,opt.samples,opt.batchSize do
      -- Set noise depending on the type
      if opt.noisetype == 'uniform' then
          noise:uniform(-1, 1)
      elseif opt.noisetype == 'normal' then
          noise:normal(0, 1)
      end
      
      -- Generate images (number specified by opt.batchSize)
      local imageSet = net:forward(noise)

      -- Save images and update dataset file with image path and its vector noise
      saveData(imageSet, noise, outFile, opt.outputFolder, i)
      print(string.format("%d/%d",torch.ceil(i/opt.batchSize),torch.ceil(opt.samples/opt.batchSize)))
  end
  
  -- Store Lua table with all the data
  local referenced = false
  torch.save(opt.outputFolder..'groundtruth.dmp', outFile, opt.format, referenced)
 
  print('Done!')
end

main()