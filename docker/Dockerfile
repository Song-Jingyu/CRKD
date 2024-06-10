FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH
ENV NVIDIA_DRIVER_CAPABILITIES=all

# add libglvnd support (More info: https://hub.docker.com/r/nvidia/opengl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \ 
        unzip \
        ffmpeg \
        libsm6 \
        libxext6 \
        tmux

# python base
RUN apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        python3

# BEVFusion base
RUN apt-get install build-essential g++ gcc -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# make python modules in /root/.local visible to non-root users
RUN find /root -type d -exec chmod 755 {} +

RUN rm -rf /tmp/* /var/lib/apt/lists/*
