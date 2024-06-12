FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get remove -y python3 && apt autoremove -y && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    ca-certificates \
    curl \
    wget \
    unzip \
    rustc \
    cargo \
    git \
    libatlas-base-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3 and python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

# get pip without grabbing python3.8 as a dependency
WORKDIR /build
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install --upgrade pip

WORKDIR /app

COPY setup.py .
COPY requirements.txt .
COPY README.md .
COPY template/__init__.py ./template/__init__.py
RUN python -m pip install uv
RUN python -m uv pip install -r requirements.txt --prerelease=allow --no-cache-dir
RUN python -m pip install -e .
COPY . .

ENV HF_HOME=checkpoints

CMD ["/bin/bash"]
