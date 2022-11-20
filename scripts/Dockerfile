FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS venv

LABEL maintainer "DECI.AI <services@deci.ai>"

ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /SG

RUN apt-get update && apt-get install -y python3-pip python3-venv python-is-python3 pip libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --copies /SG/venv
RUN . /SG/venv/bin/activate && pip install git+https://github.com/Deci-AI/super-gradients.git@stable --no-cache-dir


FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 AS prod

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 python-is-python3 python3-distutils python3-typing-extensions \
    && rm -rf /var/lib/apt/lists/*

#copying the venv and making it available as default
COPY --from=venv /SG/venv /SG/venv
ENV PATH /SG/venv/bin:$PATH


WORKDIR /SG

CMD bash