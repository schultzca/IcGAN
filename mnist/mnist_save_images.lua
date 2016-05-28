image = require 'image'
mnist = require 'mnist'
trainset = mnist.traindataset()
for i = 1, trainset.size do
  image.save('images/'..i..'.png', trainset.data[i])
end