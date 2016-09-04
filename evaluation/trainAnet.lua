require 'image'
require 'nn'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')

local function getParameters()
  local opt = {
        name = 'Anet_celebA',
        batchSize = 128,
        outputPath= 'checkpoints/',        -- path used to store the Anet network
        datasetPath = 'celebA/', -- folder where the dataset is stored (not the file itself)
        split = 0.66,           -- split between train and test (i.e 0.66 -> 66% train, 33% test)
        nConvLayers = 4,        -- # of convolutional layers on the net
        nf = 32,                -- #  of filters in hidden layer
        nEpochs = 15,           -- #  of epochs
        lr = 0.0001,            -- initial learning rate for adam
        beta1 = 0.1,            -- momentum term of adam
        display = 1,            -- display 1= train and test error, 2 = error + batches images, 0 = false
        gpu = 1                 -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
        
  }
  
  for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
  
  if opt.display then require 'display' end
  
  return opt
end

local function readDataset(path)
-- There's expected to find in path a file named images.dmp and imLabels.dmp
-- which contains the images X and attribute vectors Y.
    print('Loading X from '..path..'images.dmp') -- Change to images.dmp
    local X = torch.load(path..'images.dmp')
    print(('Done. Loaded %.2f GB (%d images).'):format((4*X:size(1)*X:size(2)*X:size(3)*X:size(4))/2^30, X:size(1)))
    X:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
    
    print('Loading Y from '..path..'imLabels.dmp')
    local Y = torch.load(path..'imLabels.dmp')
    print(('Done. Loaded %d attributes'):format(Y:size(1)))
    
    return X, Y
end

local function splitTrainTest(x, y, split)
    local xTrain, yTrain, xTest, yTest
    
    local nSamples = x:size(1)
    local splitInd = torch.floor(split*nSamples) --182638
    
    xTrain = x[{{1,splitInd},{},{},{}}]
    yTrain = y[{{1,splitInd},{}}]
    
    xTest = x[{{splitInd+1,nSamples},{},{},{}}]
    yTest = y[{{splitInd+1,nSamples},{}}]
    
    return xTrain, yTrain, xTest, yTest
end

local function getNetwork(sample, nFiltersBase, outputSz, nConvLayers)
  -- Encoder architecture taken from Autoencoding beyond pixels using a learned similarity metric (VAE/GAN hybrid)
  
  -- Sample is used to know the dimensionality of the data. 
  -- For convolutional layers we are only interested in the third dimension (RGB or grayscale)
    local inputSize = sample:size(1)
    local encoder = nn.Sequential()
    local FCsize = 512
    -- Assuming nFiltersBase = 64, nConvLayers = 3
    -- 1st Conv layer: 5×5 64 conv. ↓, BNorm, ReLU
    --           Data: 32x32 -> 16x16
    encoder:add(nn.SpatialConvolution(inputSize, nFiltersBase, 5, 5, 2, 2, 2, 2))
    encoder:add(nn.SpatialBatchNormalization(nFiltersBase))
    encoder:add(nn.ReLU(true))
    
    -- 2nd Conv layer: 5×5 128 conv. ↓, BNorm, ReLU
    --           Data: 16x16 -> 8x8
    -- 3rd Conv layer: 5×5 256 conv. ↓, BNorm, ReLU
    --           Data: 8x8 -> 4x4
    local nFilters = nFiltersBase
    for j=2,nConvLayers do
        encoder:add(nn.SpatialConvolution(nFilters, nFilters*2, 5, 5, 2, 2, 2, 2))
        encoder:add(nn.SpatialBatchNormalization(nFilters*2))
        encoder:add(nn.ReLU(true))
        nFilters = nFilters * 2
    end
    
    
     -- 4th FC layer: 2048 fully-connected
    --         Data: 4x4 -> 16
    encoder:add(nn.View(-1):setNumInputDims(3)) -- reshape data to 2d tensor (samples x the rest)
    -- Assuming squared images and conv layers configuration (kernel, stride and padding) is not changed:
    --nFilterFC = (imageSize/2^nConvLayers)²*nFiltersLastConvNet
    local inputFilterFC = (sample:size(2)/2^nConvLayers)^2*nFilters
    encoder:add(nn.Linear(inputFilterFC, FCsize)) 
    encoder:add(nn.BatchNormalization(FCsize))
    encoder:add(nn.ReLU(true))
    local outputSize
    encoder:add(nn.Linear(FCsize, outputSz))

    local criterion = nn.MSECriterion()
    
    return encoder, criterion
end

local function assignBatches(batchX, batchY, x, y, batch, batchSize, shuffle)
    
    data_tm:reset(); data_tm:resume()
    batchX:copy(x:index(1, shuffle))
    batchY:copy(y:index(1, shuffle))
    data_tm:stop()
    
    return batchX, batchY
end

local function displayConfig(disp, title)
    -- initialize error display configuration
    local errorData, errorDispConfig
    if disp then
        errorData = {}
        errorDispConfig =
          {
            title = 'Anet error - ' .. title,
            win = 1,
            labels = {'Epoch', 'Train error', 'Test error'},
            ylabel = "Error",
            legend='always'
          }
    end
    return errorData, errorDispConfig
end

function main()

  local opt = getParameters()
  if opt.display then display = require 'display' end
  
  -- Set timers
  local epoch_tm = torch.Timer()
  local tm = torch.Timer()
  local data_tm = torch.Timer()

  -- Read dataset
  local X, Y
  X, Y = readDataset(opt.datasetPath)
  
  -- Split train and test
  local xTrain, yTrain, xTest, yTest
  xTrain, yTrain, xTest, yTest = splitTrainTest(X, Y, opt.split)

  -- X: #samples x im3 x im2 x im1
  -- Y: #samples x ny
  
  -- Set network architecture
  local Anet, criterion = getNetwork(xTrain[1], opt.nf, yTrain:size(2), opt.nConvLayers)
 
  -- Initialize batches
  local batchX = torch.Tensor(opt.batchSize, xTrain:size(2), xTrain:size(3), xTrain:size(4))
  local batchY = torch.Tensor(opt.batchSize, yTrain:size(2))
  
  -- Copy variables to GPU
  if opt.gpu > 0 then
     require 'cunn'
     cutorch.setDevice(opt.gpu)
     batchX = batchX:cuda();  batchY = batchY:cuda();
     
     if pcall(require, 'cudnn') then
        require 'cudnn'
        cudnn.benchmark = true
        cudnn.convert(Anet, cudnn)
     end
     
     Anet:cuda()
     criterion:cuda()
  end
  
  local params, gradParams = Anet:getParameters() -- This has to be always done after cuda call
  
  -- Define optim (general optimizer)
  local errorTrain
  local errorTest
  local function optimFunction(params) -- This function needs to be declared here to avoid using global variables.
      -- reset gradients (gradients are always accumulated, to accommodat batch methods)
      gradParams:zero()
      
      local outputs = Anet:forward(batchX)
      errorTrain = criterion:forward(outputs, batchY)
      local dloss_doutput = criterion:backward(outputs, batchY)
      Anet:backward(batchX, dloss_doutput)
      
      return errorTrain, gradParams
  end
  
  local optimState = {
     learningRate = opt.lr,
     beta1 = opt.beta1,
  }
  
  local nTrainSamples = xTrain:size(1)
  local nTestSamples = xTest:size(1)
  
  -- Initialize display configuration (if enabled)
  local errorData, errorDispConfig = displayConfig(opt.display, opt.name)
  
  paths.mkdir(opt.outputPath)
  
  -- Train network
  local batchIterations = 0 -- for display purposes only
  for epoch = 1, opt.nEpochs do
      epoch_tm:reset()
      local shuffle = torch.randperm(nTrainSamples):long()
      for batch = 1, nTrainSamples-opt.batchSize+1, opt.batchSize  do
          tm:reset()
          -- Assign batches
          --[[local splitInd = math.min(batch+opt.batchSize, nTrainSamples)
          batchX:copy(xTrain[{{batch,splitInd}}])
          batchY:copy(yTrain[{{batch,splitInd}}])--]]
          
          batchX, batchY = assignBatches(batchX, batchY, xTrain, yTrain, batch, opt.batchSize, shuffle)
          
          if opt.display == 2 and batchIterations % 20 == 0 then
              display.image(image.toDisplayTensor(batchX,0,torch.round(math.sqrt(opt.batchSize))), {win=2, title='Train mini-batch'})
          end
          
          -- Update network
          optim.adam(optimFunction, params, optimState)
          
          -- Display train and test error
          if opt.display and batchIterations % 20 == 0 then
              -- Test error
              batchX, batchY = assignBatches(batchX, batchY, xTest, yTest, torch.random(1,nTestSamples-opt.batchSize+1), opt.batchSize, torch.randperm(nTestSamples))
              local outputs = Anet:forward(batchX)
              errorTest = criterion:forward(outputs, batchY)
              table.insert(errorData,
              {
                batchIterations/math.ceil(nTrainSamples / opt.batchSize), -- x-axis
                errorTrain, -- y-axis for label1
                errorTest -- y-axis for label2
              })
              display.plot(errorData, errorDispConfig)
              if opt.display == 2 then
                  display.image(image.toDisplayTensor(batchX,0,torch.round(math.sqrt(opt.batchSize))), {win=3, title='Test mini-batch'})
              end
          end
          
          -- Verbose
          if ((batch-1) / opt.batchSize) % 1 == 0 then
             print(('Epoch: [%d][%4d / %4d]  Error (train): %.4f  Error (test): %.4f  '
                       .. '  Time: %.3f s  Data time: %.3f s'):format(
                     epoch, ((batch-1) / opt.batchSize),
                     math.ceil(nTrainSamples / opt.batchSize),
                     errorTrain and errorTrain or -1,
                     errorTest and errorTest or -1,
                     tm:time().real, data_tm:time().real))
         end
         batchIterations = batchIterations + 1
      end
      print(('End of epoch %d / %d \t Time Taken: %.3f s'):format(
            epoch, opt.nEpochs, epoch_tm:time().real))
            
      -- Store network
      torch.save(opt.outputPath .. opt.name .. '_' .. epoch .. 'epochs.t7', Anet:clearState())
      torch.save('checkpoints/' .. opt.name .. '_error.t7', errorData)
  end
  
end

main()
--os.execute("poweroff")