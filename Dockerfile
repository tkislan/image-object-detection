FROM tensorflow/tensorflow:1.15.2 as amd

ARG MODEL_URL

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/project/app:/opt/tensorflow_models/research:/opt/tensorflow_models/research/slim

RUN apt-get update && apt-get install -y curl git protobuf-compiler libfreetype6-dev
RUN pip3 install Matplotlib==3.2.1 Pillow==7.1.2 minio==5.0.10
RUN git clone https://github.com/tensorflow/models /opt/tensorflow_models
RUN cd /opt/tensorflow_models/research && \
    git checkout v1.13.0 && \
    protoc object_detection/protos/*.proto --python_out=. && \
    python3 setup.py install

RUN mkdir -p /tmp/model/download && \
    cd /tmp/model/download && \
    curl -L -O "${MODEL_URL}" && \
    mkdir -p /tmp/model/files && \
    tar -xzvf /tmp/model/download/*.tar.gz -C /tmp/model/files

RUN apt-get install -y python3-opencv

ADD image_object_detection /opt/project/app/image_object_detection

RUN mkdir -p /opt/models/
RUN python3 \
        /opt/project/app/image_object_detection/utils/trt_model/trt_model_convert.py \
        "$(echo /tmp/model/files/*)/pipeline.config" \
        "$(echo /tmp/model/files/*)/model.ckpt" && \
    mv trt_graph.pb /opt/models/trt_graph.pb
RUN cp -v "$(echo /tmp/model/files/*)/frozen_inference_graph.pb" /opt/models/graph.pb
RUN ls -alh /opt/models

WORKDIR /opt/project/app

FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.2-tf1.15-py3

RUN apt-get update && apt-get install -y git protobuf-compiler libfreetype6-dev
RUN pip3 install Matplotlib==3.2.1 Pillow==7.1.2 minio==5.0.10
RUN git clone https://github.com/tensorflow/models /opt/tensorflow_models
RUN cd /opt/tensorflow_models/research && \
    git checkout v1.13.0 && \
    protoc object_detection/protos/*.proto --python_out=. && \
    python3 setup.py install

RUN apt-get install -y python3-opencv

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/project/app:/opt/tensorflow_models/research:/opt/tensorflow_models/research/slim

ADD image_object_detection /opt/project/app/image_object_detection

COPY --from=amd /opt/models /opt/models

WORKDIR /opt/project/app

CMD python3 image_object_detection/app.py
