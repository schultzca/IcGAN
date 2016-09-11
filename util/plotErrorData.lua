local disp = require 'display'

local type = 1 -- 1: GAN error, 2: encoder error
local path = './checkpoints/error.t7'
local errorDispConfig
if type == 1 then
    errorDispConfig =
          {
            title = 'Generator and discriminator error',
            labels = {'Batch iterations', 'G error', 'D error'},
            ylabel = "G error",
            y2label = "D error",
            legend = 'always',
            axes = { y2 = {valueRange = {0,1}}},
            series = {
              ['D error'] = { axis = 'y2' }
            }
          }
else -- encoder error
    errorDispConfig =
              {
                title = 'Encoder error',
                labels = {'Batch iterations', 'Train error', 'Test error'},
                ylabel = "Error",
                legend='always'
              }
end

local errorData = torch.load(path)
disp.plot(errorData, errorDispConfig)