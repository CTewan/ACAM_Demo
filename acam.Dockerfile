FROM nvidia/cuda:10.0-cudnn7-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
		build-essential \
    pkg-config 
		wget \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

COPY ./ /app

RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN ./bin/protoc object_detection/models/research/object_detection/protos/*.proto --python_out=.

RUN apt-get install -y python3-pip
RUN pip install -r requirements.txt