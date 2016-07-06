local util = {}

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
          noises:normal(0, 1)
      end
      
      -- Generate images
      net:forward(noises)
  
  end
end

