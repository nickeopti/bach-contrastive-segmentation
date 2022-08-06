FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install build-essential gdb lcov pkg-config \
      libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
      libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
      lzma lzma-dev tk-dev uuid-dev zlib1g-dev wget -y

RUN wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz
RUN tar -xf Python-3.10.6.tgz
RUN cd Python-3.10.6 && ./configure --enable-optimizations && make install

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt install openslide-tools -y

WORKDIR /home/code
COPY src src/
COPY pyproject.toml .
COPY setup.py .
RUN pip3 install -e .

WORKDIR /home
