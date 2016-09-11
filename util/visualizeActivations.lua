require 'cudnn'
require 'cunn'
require 'image'
local display = require 'display'
local net = torch.load('./checkpoints/c_celebA_64_filt_Yconv1_noTest_25_net_G.t7')
net:evaluate()
torch.manualSeed(1)
local Z = torch.Tensor(1,100,1,1):normal(0,1):cuda()
local Y = torch.Tensor(1,18):fill(-1):cuda()
Y[{{1},{7}}] = -1 -- Activate eyeglasses
net:forward({Z,Y})

local plotInfo = {}
  
local moduleList = {2, 5, 8, 11, 14}
local j = 1
for _,i in pairs(moduleList) do
    local w = net:get(i).weight    
    local o = net:get(i).output
    local name = torch.type(net:get(i))
    
    w = w:view(w:size(1)*w:size(2), w:size(3), w:size(4))
    o = o:view(o:size(2), o:size(3), o:size(4))
    
    plotInfo.title = name..' '..i..' weight'
    plotInfo.win = j
    --display.image(image.toDisplayTensor{input=w,padding=0,nrow=math.ceil(math.sqrt(w:size(1))),scaleeach=true}, plotInfo)
    
    plotInfo.title = name..' '..i..' output'
    plotInfo.win = j+1
    display.image(image.toDisplayTensor{input=o,padding=0,nrow=math.ceil(math.sqrt(o:size(1))),scaleeach=true}, plotInfo)
    
    j = j + 2
    
end


