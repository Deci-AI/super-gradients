ARG DOCKER_IMAGE_TAG=11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:${DOCKER_IMAGE_TAG}

LABEL maintainer "DECI.AI <services@deci.ai>"
ARG SG_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /SG

RUN apt-get update && apt-get install -y python3-pip python-is-python3 pip libgl1 libglib2.0-0 git python3-distutils python3-typing-extensions \
    && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/Deci-AI/super-gradients.git@${SG_VERSION} --no-cache-dir && pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir 

WORKDIR /SG

CMD bash