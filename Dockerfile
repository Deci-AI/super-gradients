FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 as base_container
ENV INSTALL_PATH /workspace
ENV VENV_NAME venv
ENV PATH=${INSTALL_PATH}/${VENV_NAME}/bin:${PATH}

ARG SG_VERSION="3.0.7"
ARG PIP_FLAGS="--no-deps"
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION="3.8"
ARG PIP_CONFIG_FILE_HOST="./.config/pip/pip.conf"

# Preparing system requirements
RUN apt update && apt install -y --no-install-recommends software-properties-common &&  \
    add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y --no-install-recommends  \
    curl \
    less \
    gzip \
    sudo \
    wget \
    apt-transport-https \
    unzip \
    jq \
    git \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6 \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python-dev \
    libxml2-dev \
    libxslt-dev \
    python3-dev \
    gcc \
    vim \
    nano \
    python3-pip && apt clean && rm -rf /var/lib/apt/lists/*

# working directory
WORKDIR ${INSTALL_PATH}

# creating virtual environment (to allow multi stage buikd in the future)
RUN python${PYTHON_VERSION}  -m venv --system-site-packages --copies ${INSTALL_PATH}/${VENV_NAME}
RUN . ${INSTALL_PATH}/${VENV_NAME}/bin/activate

RUN pip install --no-cache-dir --upgrade git+https://github.com/Deci-AI/super-gradients@master
RUN pip uninstall super-gradients -y
ARG CACHEBUST=1

RUN pip install --no-cache-dir --upgrade git+https://github.com/Deci-AI/super-gradients@eugene/debug

ENV FILE_LOG_LEVEL    DEBUG
ENV CONSOLE_LOG_LEVEL DEBUG
ENV LOG_LEVEL         DEBUG

# we should do multistage build in the future

# FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 as lean_image #todo
#ENV INSTALL_PATH /workspace
#ENV VENV_NAME venv
#ENV PATH=${INSTALL_PATH}/${VENV_NAME}/bin:${PATH}
#WORKDIR ${INSTALL_PATH}
#COPY --from=base_container $INSTALL_PATH $INSTALL_PATH #todo

# Coping the content to deci-algo to be available in the container
WORKDIR ${INSTALL_PATH}
