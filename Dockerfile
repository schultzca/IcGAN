FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Charles Schultz <schultz.ca@gmail.com>

# Install software dependencies
RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  ipython3 \
  libssl-dev \
  libzmq3-dev \
  python-zmq \
  python-pip \
  sudo

# Clone torch repository
RUN git clone https://github.com/torch/distro.git /root/torch --recursive

# Install torch
WORKDIR /root/torch
RUN bash install-deps
RUN bash install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install torch dependencies
RUN luarocks install optnet
RUN luarocks install cudnn
RUN luarocks install threads
RUN luarocks install display

# Copy IcGAN
WORKDIR /app
ADD . /app

# Extract pre-trained models
WORKDIR /app/generation/models
RUN tar xvzf celeba_pretrained_models.tar.gz

WORKDIR /app

CMD ["/bin/bash"]