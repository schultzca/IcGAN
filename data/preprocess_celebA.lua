require 'torch'
require 'image'
require 'nn'
require 'io'

local trainSet = {}
local dataroot = os.getenv('DATA_ROOT')
if dataroot == nil then
    dataroot = 'celebA'
end
local data = dataroot .. '/img_align_celeba'

-- Create filter
local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
local function isImage(filename) 
    -- Filter all files with no image extension
    for i=1, #extensionList do
        if filename:find(i) then return true end
    end
    return false
end

local imNames = {}

-- Get raw paths (including non-image paths)
local fileIterator = io.popen("ls "..data)

for filename in fileIterator:lines() do
    if isImage(filename) then
        imNames[#imNames+1] = filename
    end
end

local outSz = 64
-- Open first image to get its sizes
local im = image.load(data.. '/' .. imNames[1]):float()

-- Initialize output container
local images = torch.FloatTensor(#imNames, im:size(1), outSz, outSz)

local x1, y1 = 30, 40
for i, name in ipairs(imNames) do
    im = image.load(data.. '/' .. imNames[i], im:size(1), 'float')
    im = image.crop(im, x1, y1, x1 + 138, y1 + 138)
    im = image.scale(im, outSz, outSz)
    image.save(data.. '/' ..imNames[i], im)
    images[{{i},{},{},{}}] = im
    print(i)
end

print('Saving...')
torch.save(dataroot..'/images.dmp', images)
torch.save(dataroot..'/imNames.dmp', imNames)
print('Done!')
