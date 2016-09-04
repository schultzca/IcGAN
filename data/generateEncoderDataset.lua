-- This file loads an already trained GAN and generate a dataset that
-- consists of generated images and the input noise Z that generated them.
-- This dataset will be later used for training an encoding GAN (given an image,
-- encode it as a noise vector Z so as the decoding GAN can generate the same image
-- from it).

require 'image'
require 'nn'
require "lfs"
require "io"
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')
--torch.manualSeed(123) -- This only works if gpu == 0

local function getParameters()
  local opt = {
    samples = 202599,          -- total number of samples to generate
    batchSize = 256,         -- number of samples to produce at the same time
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = 'checkpoints/c_celebA_64_filt_Yconv1_noTest_wrongYFixed_24_net_G.t7',-- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px,
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,              -- size of noise vector
    outputFolder = 'celebA/c_Yconv1_generatedDataset/', -- path where the dataset will be stored
    outputFormat = 'binary', -- (binary | ascii) binary is faster, but platform-dependent.
    storeAsTensor = true,    -- true -> store images as tensor, false -> store images as images (lossy)
    -- Conditional GAN parameters
    dataset = 'celebA',     -- mnist | celebA  
  }
  
  for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
  
  assert(opt.samples >= opt.batchSize, "Batch size (opt.batchSize) can't be greater than number of samples (opt.samples).")
  
  if not opt.storeAsTensor then
      print("Warning: creating dataset with storeAsTensor == false. Storing as images instead of tensor is a lossy process.")
  end
  
  if opt.gpu > 0 then
      require 'cunn'
      require 'cudnn'
  end
  
  return opt
end

local function sampleYceleba(Y, idx, batchSize, imLabels)
    local modIdx = ((idx-1)%imLabels:size(1))+1
    local endIdx = modIdx+batchSize-1
    if endIdx > imLabels:size(1) then
        -- Fill the rest of Y
        Y:copy(torch.cat(imLabels[{{modIdx,imLabels:size(1)},{}}], imLabels[{{1,endIdx%imLabels:size(1)},{}}],1))
    else
        Y:copy(imLabels[{{modIdx,endIdx},{}}])
    end
end

local function readCelebaLabels(labelPath, nSamples)
    -- Index of the attributes from celebA that will be ignored
    local attrFil = {1,2,3,4,7,8,11,14,15,17,20,24,25,26,28,30,31,35,37,38,39,40}
    local celebaSize = 202599
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
    
    local imLabels = torch.IntTensor(celebaSize, #labelHeader)
    
    local ySize = #labelHeader
    
    -- Read the rest of the file
    local i = 1
    local skippedImages = 0 -- Controls if some images are missing
    for line in file:lines() do  
      local l = line:split('%s+')
      -- Check if image in labels file exists on imPaths
      -- If not, skip it from imLabels
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
      i = i + 1
    end
    
    local y = torch.Tensor(nSamples, ySize)
    
    -- We subtract 19961 (test set size) as not to use y from the test set
    local randIdx = torch.randperm(celebaSize-19961):narrow(1,1,nSamples)
    y = imLabels:index(1, randIdx:long())
    
    return y
end


local function stabilizeBN(net, input, noiseType, celebaLabels)
  -- When using a net in evaluation mode, running_mean and running_var
  -- from BN layers might be a bit off from training, which yields to major
  -- differences between images generated on training mode from evaluation mode.
  -- To avoid this, we do some mini-batches iterations without training so that
  -- they are re-estimated, and then we save the network again.
  for i=1,100 do
      -- Set noise depending on the type
      if noiseType == 'uniform' then
          input[1]:uniform(-1, 1)
      elseif noiseType == 'normal' then
          input[1]:normal(0, 1)
      end
      
      if celebaLabels ~= nil then
          sampleYceleba(input[2], i, input[1]:size(1), celebaLabels)
      end
      
      -- Generate images
      net:forward(input)
  end
  
  net.__BNrefreshed = true
  --net:evaluate()
end

local function initializeNet(net, opt)
  -- Initializes the net and the input noise.
  -- This involves passing the network to GPU and other optimizations.
  local imLabels
  local Z = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
  local Y
  if opt.dataset == 'mnist' then
      opt.ny = 10 -- Y length
      Y = torch.Tensor(opt.batchSize, opt.ny):fill(-1)
      for i=1,opt.batchSize do
        Y[{{i},{((i-1)%opt.ny)+1}}] = 1 -- Y specific to MNIST dataset
      end
  elseif opt.dataset == 'celebA' then
      --Read label dataset and pick randomly n Y vectors
      imLabels = readCelebaLabels(opt.dataset..'/list_attr_celeba.txt', opt.samples)
      opt.ny = imLabels:size(2)
      Y = torch.zeros(opt.batchSize, opt.ny)
  end
  
  if opt.gpu > 0 then
      net = net:cuda()
      cudnn.convert(net, cudnn)
      Z = Z:cuda(); Y = Y:cuda()
  else
      net:float()
  end
  local sampleInput = {Z:narrow(1,1,2), Y:narrow(1,1,2)}
  optnet.optimizeMemory(net, sampleInput)
  
  -- Refresh mean and std from BN layers. This needs to be done at least once after training.
  if not net.__BNrefreshed then
      -- There's no need to do this more than once. Once the network with
      -- the refreshed BN is saved, net.__BNrefreshed will indicate whether
      -- its BN has been refreshed or not.
      print('Refreshing BN...')
      stabilizeBN(net,{Z,Y},'normal',imLabels)
      torch.save(opt.net, net)
  end
    
  -- Put network to evaluate mode so batch normalization layers 
  -- do not change its parameters on test time.
  net:evaluate()
  
  return net, Z, Y, imLabels
end

local function createFolder(path)
    local state, error
    
    state, error = lfs.mkdir(path)
    assert(state or error == 'File exists',"Couldn't create directory "..path) 
end

local function initializeOutputFile(opt, nSamplesReal, net)
  local outFile

  if opt.storeAsTensor then
      outFile = {
        storeAsTensor = opt.storeAsTensor,
        X = nil, -- This value will be updated below
        imSize = nil, -- This value will be updated below
        Z = torch.Tensor(nSamplesReal, opt.nz, opt.imsize, opt.imsize),
        Y = torch.Tensor(nSamplesReal, opt.ny)
      }
  
  else
      outFile = {
        storeAsTensor = opt.storeAsTensor,
        relativePath = opt.outputFolder..'images/', -- write relative path once, not for every image
        imNames = {},
        imSize = nil, -- This value will be updated below
        Z = torch.Tensor(nSamplesReal, opt.nz, opt.imsize, opt.imsize),
        Y = torch.Tensor(nSamplesReal, opt.ny)
      }
  end
  
  -- Input Z to the net to know the output image size
  local sampleInput = {outFile.Z:narrow(1,1,2), outFile.Y:narrow(1,1,2)}
  if opt.gpu > 0 then 
      sampleInput[1] = sampleInput[1]:cuda() 
      sampleInput[2] = sampleInput[2]:cuda()
  end 
  local imSize = net:forward(sampleInput):size() 
  outFile.imSize = {imSize[2], imSize[3], imSize[4]}
  if opt.storeAsTensor then outFile.X = torch.Tensor(nSamplesReal, imSize[2], imSize[3], imSize[4]) end
  
  return outFile
end

local function saveData2Txt(imageSet, Z, path, idx)
-- imageSet dimension: # of images x im3 x im2 x im1
-- Z: # of images x nz x 1 x 1
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
        -- Write Z to file
        file:write(Z[{{j},{},{},{}}]:type('torch.CharTensor'), "\n")
    end

    file:close()
end

local function saveData(imageSet, Z, Y, outFile, path, storeAsTensor, idx)
-- imageSet dimension: # of images x im3 x im2 x im1
-- Z: # of images x nz x 1 x 1
-- Y: # of images x ny
-- path: folder where the images will be stored
-- idx: base index to know where to store things in outFile
    
    if not storeAsTensor then
        -- Store as images
        for j=1,imageSet:size(1) do
            local i = idx+j-1
            local imageName = string.format("%.7d.png",i)
            local imagePath = path..'images/'..imageName
            local im = imageSet[{{j},{},{},{}}]
            -- Save image
            im:resize(im:size(2), im:size(3), im:size(4)) -- 4 to 3 dimensions
            image.save(imagePath, im)
            -- Update output file with image paths
            outFile.imNames[i] = imageName
        end
    else
        -- Update output file with image tensors
        outFile.X[{{idx,idx+imageSet:size(1)-1},{},{},{}}] = imageSet:float() -- float() is necessary in case Z is in GPU
    end
    -- Update output file with Z
    outFile.Z[{{idx,idx+imageSet:size(1)-1},{},{},{}}] = Z:float() -- float() is necessary in case Z is in GPU
    outFile.Y[{{idx,idx+imageSet:size(1)-1},{}}] = Y:float()
  
end

function main()

  local opt = getParameters()
  
  -- Load generative network
  local net = torch.load(opt.net)
  local imLabels -- only used in celebA
  local Z, Y
  
  net, Z, Y, imLabels = initializeNet(net, opt)
  
  -- Create output folder path and other subfolders
  createFolder(opt.outputFolder)
  if not opt.storeAsTensor then  createFolder(opt.outputFolder..'images/') end
  
  -- Note: If opt.samples is not divisble by opt.batchSize,
  -- more samples than opt.samples will be produced.
  local nSamplesReal = math.floor(opt.samples/opt.batchSize)*opt.batchSize
  if nSamplesReal ~= opt.samples then
      nSamplesReal = nSamplesReal + opt.batchSize%opt.samples
  end

  -- Initialize output file that will contain all the information
  local outFile = initializeOutputFile(opt, nSamplesReal, net)
  
  -- Create as many pairs of generated image and {Z, Y} vector
  -- as specified by opt.samples.
  local imageSet
  for i=1,opt.samples,opt.batchSize do
      -- Set Z depending on the type
      if opt.noisetype == 'uniform' then
          Z:uniform(-1, 1)
      elseif opt.noisetype == 'normal' then
          Z:normal(0, 1)
      end
      
      -- Y sampling is needed for celebA, as unlike MNIST, there are many possible combinations
      if opt.dataset == 'celebA' then
          sampleYceleba(Y,i,opt.batchSize,imLabels)
      end
      
      -- Generate images (number specified by opt.batchSize)
      imageSet = net:forward{Z,Y}
      
      -- Save images and update dataset file with image path and its vector Z
      saveData(imageSet, Z, Y, outFile, opt.outputFolder, opt.storeAsTensor, i)
      print(string.format("%d/%d",torch.ceil(i/opt.batchSize),torch.ceil(opt.samples/opt.batchSize)))
  end

  
  -- Store Lua table with all the data
  local referenced = false
  torch.save(opt.outputFolder..'groundtruth.dmp', outFile, opt.format, referenced)
 
  print('Done!')
end

main()