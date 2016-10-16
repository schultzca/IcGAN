-- This code is used to create a dataset of reconstructed images from the test set.
-- Given a path to a file containing the images X from the dataset and its real labels Y,
-- it encodes and decodes those images and creates a file with the reconstructed images X'
-- and the original labels Y. Then, this dataset will be used in evaluateModel.lua to evaluate
-- whether or not the reconstructed images keep the original attribute information.

require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
require 'lfs'
torch.setdefaulttensortype('torch.FloatTensor')

local function getParameters()
  local opt = {
      batchSize = 18,
      decNet = 'checkpoints/c_celebA_64_filt_Yconv1_noTest_wrongYFixed_24_net_G.t7',--c_celebA_64_filt_Yconv1_noTest_wrongYFixed_24_net_G, --c_mnist_-1_1_25_net_G.t7,-- path to the generator network
      encZnet = 'checkpoints/encoderZ_c_celeba_7epochs.t7',--'checkpoints/encoder_c_mnist_6epochs.t7' 'checkpoints/encoder128Filters2FC_dataset2_2_6epochs.t7',
      encYnet = 'checkpoints/Anet2_celebA_5epochs.t7',
      gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
      display = 0,
      nz = 100,
      path = 'celebA/',
      outputFolder = 'celebA/reconstructedDataset/', -- path where the dataset will be stored
      threshold = true, -- threshold Y vectors to binary or not
  }
  
  if opt.gpu > 0 then
      require 'cunn'
      require 'cudnn'
  end
  return opt
end

local function readDataset(path)
-- There's expected to find in path a file named im_and_labels_test_set.dmp
-- which contains *test* images X and their attribute vectors Y.

  print('Loading '..path..'im_and_labels_test_set.dmp')
  local data = torch.load(path..'im_and_labels_test_set.dmp')
  local X = data.X
  local Y = data.Y
  print(('Done. Loaded %.2f GB (%d images).'):format((4*X:size(1)*X:size(2)*X:size(3)*X:size(4))/2^30, X:size(1)))
  
  -- Check images are in range [-1, 1]
  if X:min() >= 0 then 
      X:mul(2):add(-1) -- make it [0, 1] -> [-1, 1] 
  end
  return X, Y
end

local function createFolder(path)
    local state, error
    
    state, error = lfs.mkdir(path)
    assert(state or error == 'File exists',"Couldn't create directory "..path) 
end

function main()

  local opt = getParameters()
  
  -- Create output folder
  createFolder(opt.outputFolder)
  
  -- Load nets
  local dec = torch.load(opt.decNet)
  local encZ = torch.load(opt.encZnet) 
  local encY = torch.load(opt.encYnet)
  
  -- Load data
  local inputX, Y = readDataset(opt.path)
  local outX = inputX:clone():zero()
  local ny = Y:size(2)-- Y label length. This depends on the dataset.
  local nSamples = inputX:size(1)
  
  -- Check if number of samples is multiple of batch size
  if (nSamples % opt.batchSize) ~= 0 then
      print(('Warning: batch size %d is not multiple of the total number of samples %d. Last %d samples will be ignored.'):format(opt.batchSize, nSamples, nSamples % opt.batchSize))
      outX = torch.Tensor(nSamples-(nSamples % opt.batchSize), outX:size(2), outX:size(3), outX:size(4))
  end
  
  -- Initialize batch
  local batchX = torch.Tensor(opt.batchSize, inputX:size(2), inputX:size(3), inputX:size(4))
    
  -- CPU to GPU  
  if opt.gpu > 0 then
     cudnn.convert(dec, cudnn)
     dec:cuda()
     encZ:cuda()
     encY:cuda()
     batchX = batchX:cuda()
  else
     dec:float()
     encZ:float()
     encY:float()
  end
  
  dec:evaluate()
  encZ:evaluate()
  encY:evaluate()
   
  -- Initiate main loop
  for batch = 1, nSamples-opt.batchSize+1, opt.batchSize  do
      -- Assign batch
      batchX:copy(inputX[{{batch,batch+opt.batchSize-1},{},{},{}}])
      
      -- Encode images X to attribute vectors Y with encY
      local tmpY = encY:forward(batchX)
      
      -- Encode images X to a latent representation Z with encZ
      local tmpZ = encZ:forward(batchX)
      
      -- Adapt dimensionality of Z
      tmpZ:resize(tmpZ:size(1), tmpZ:size(2), 1, 1)
      
      -- Decode images
      local tmp = dec:forward{tmpZ, tmpY}
      
      -- Display (optional)
      if opt.display == 1 then disp.image(tmp,{win=0}) end
      
      -- Copy reconstructed images to CPU
      outX[{{batch,batch+opt.batchSize-1}}]:copy(tmp)

      print(("%4d / %4d"):format(math.floor(batch / opt.batchSize), math.floor(nSamples / opt.batchSize)))
  end
  
  -- Save output file
  local outputData = {}
  outputData.X = outX
  outputData.Y = Y
  
  torch.save(opt.outputFolder..'groundtruth.dmp', outputData)
  
  print('Done.')
  
end

main()