FROM nvidia/cuda:11.7.1-base-ubuntu20.04

LABEL maintainer "DECI.AI <services@deci.ai>"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install super-gradients

RUN mkdir /SG
WORKDIR /SG
