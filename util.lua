local util = {}

function util.save(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
	-- TODO: figure out why giant storage-offsets being created on typecast
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output = netsave.output.new()
    netsave.gradInput = netsave.gradInput.new()

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end

function util.cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
						 l.kW, l.kH, l.dW, l.dH, 
						 l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end

-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed to generate samples.
function util.optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end

function util.stabilizeBN(net, noises, noiseType)
  -- When using a net in evaluation mode, running_meand and running_var
  -- from BN layers might be a bit off from training, which yields to major
  -- differences between images generated on training mode from evaluation mode.
  -- To avoid this, we do some mini-batches iterations without training so that
  -- they are re-estimated, and then we save the network again.
  for i=1,100 do
      -- Set noise depending on the type
      if noiseType == 'uniform' then
          noises:uniform(-1, 1)
      elseif noiseType == 'normal' then
          noises:normal(0, 10)
      end
      
      -- Generate images
      net:forward(noises)
  
  end
end

return util
