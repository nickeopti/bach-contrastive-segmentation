FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt update && apt install -y wget git \
    build-essential zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev curl

RUN wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz
RUN tar -xf Python-3.10.6.tgz
RUN cd Python-3.10.6 && ./configure && make install

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /home
RUN git clone https://github.com/nickeopti/bach-contrastive-segmentation.git --branch package-refactor code

RUN cd code && pip3 install -e .

COPY data data/
