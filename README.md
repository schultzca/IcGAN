Invertible Conditional GANs
===========================================

![A real image is encoded into a latent representation z and conditional information y, and then decoded into a new image. We fix z for every row, and modify y for each column to obtain variations in real samples.](https://raw.githubusercontent.com/Guim3/BcGAN/master/images/celeba_samples.png)

This is the implementation of the IcGAN model proposed in my master dissertation
from September 2016:

[*Invertible Conditional Generative Adversarial Networks.*][1] Guim Perarnau. September 2016.

The baseline used is the [Torch implementation][2] of the [DCGAN by Radford et al][3].

1. [Training the model](#1-training-the-model)
	1. [Face dataset: CelebA](#11-face-dataset-celeba)
	2. [Digit dataset: MNIST](#12-digit-dataset-mnist)
2. [Visualize the results](#2-visualize-the-results)
	

## Requisites

Please refer to [DCGAN torch repository][4] to know the requirements and dependencies to run the code.
Additionally, you will need to install the `threads` package: 

`luarocks install threads`

In order to interactively display the results, follow [these steps][7].

## 1. Training the model

The IcGAN is trained in four steps. 
1. Train the generator. 
2. Create a dataset of generated images with the generator. 
3. Train the encoder Z to map an image *x* to a latent representation *z* with the dataset generated images. 
4. Train the encoder Y to map an image *x* to a conditional information vector *y* with the dataset of real images.

All the parameters of the training phase are located in cfg/mainConfig.lua.

There is already a pre-trained model for CelebA available in case you want to skip the training part. [Here](#2-visualize-the-results) you can find instructions on how to use it.

### 1.1 Train with a face dataset: CelebA

Note: for speed purposes the whole dataset will be loaded in RAM in training time, which requires about 10 GB of RAM. Therefore, having 12 GB of RAM is a minimmum requirement. Also, the dataset will be stored as a tensor to load it faster, make sure then that you have 25 GB of free space.

#### Preprocess
`mkdir celebA; cd celebA`

Download img_align_celeba.zip from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html under the link "Align&Cropped Images".

```unzip img_align_celeba.zip; cd ..
DATA_ROOT=celebA th data/preprocess_celebA.lua```

#### Training

* Conditional GAN: parameters are already configured to run CelebA (dataset=celebA, dataRoot=celebA).
`th trainGAN.lua`
* Generate encoder dataset: 
`net=[GENERATOR_PATH] outputFolder=celebA/genDataset samples=182638 th data/generateEncoderDataset.lua`
(GENERATOR_PATH example: checkpoints/celebA_25_net_G.t7)
* Train encoder Z: `datasetPath=celebA/genDataset/ type=Z th trainEncoder.lua`
* Train encoder Y: `datasetPath=celebA/ type=Y th trainEncoder.lua`

### 1.2 Train with a digit dataset: MNIST

#### Preprocess
Download MNIST as a luarocks package: `luarocks install mnist`

#### Training

* Conditional GAN: `name=mnist dataset=mnist dataRoot=mnist th trainGAN.lua`
* Generate encoder dataset: 
`net=[GENERATOR_PATH] outputFolder=mnist/genDataset samples=60000 th data/generateEncoderDataset.lua`
(GENERATOR_PATH example: checkpoints/mnist_25_net_G.t7)
* Train encoder Z: `datasetPath=mnist/genDataset/ type=Z th trainEncoder.lua`
* Train encoder Y: `datasetPath=mnist type=Y th trainEncoder.lua`


## 2. Visualize the results

Coming soon

    

[1]: https://www.overleaf.com/read/hgvnqhyyscpc
[2]: https://github.com/soumith/dcgan.torch
[3]: https://arxiv.org/abs/1511.06434
[4]: https://github.com/soumith/dcgan.torch#prerequisites 
[5]: https://sites.google.com/site/nips2016adversarial/
[6]: link_pending
[7]: https://github.com/soumith/dcgan.torch#display-ui
