FROM python:3.5.6 as builder

RUN apt-get update && \
    apt-get install -y zip && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/
RUN pip install --no-binary pillow pillow==5.4.1

RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip \
        -o /tmp/protoc-3.6.1-linux-x86_64.zip && \
    mkdir -p /tmp/protoc && \
    unzip -q /tmp/protoc-3.6.1-linux-x86_64.zip -d /tmp/protoc && \
    mv /tmp/protoc/bin/protoc /usr/bin/protoc

RUN git clone https://github.com/tensorflow/models.git /opt/tensorflow_models && \
    (cd /opt/tensorflow_models && git reset --hard b36872b66fab34fbf71d22388709de5cd81878b4)
RUN rm -rf /opt/tensorflow_models/.git

RUN (cd /opt/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.)


FROM python:3.5.6-slim

ARG MODEL_URL

RUN apt-get update && \
    apt-get install -y libtiff5 libjpeg62-turbo zlib1g libfreetype6 liblcms2-2 libwebp6 libopenjp2-7 curl && \
    pip install tensorflow==1.12.0 pillow==5.4.1 matplotlib minio && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/ && \
    rm -r /root/.cache/pip

COPY --from=builder \
    /usr/local/lib/python3.5/site-packages/Pillow-5.4.1-py3.5.egg-info \
    /usr/local/lib/python3.5/site-packages/
COPY --from=builder \
    /usr/local/lib/python3.5/site-packages/PIL \
    /usr/local/lib/python3.5/site-packages/PIL

COPY --from=builder \
    /opt/tensorflow_models/research/object_detection \
    /opt/tensorflow_models/research/object_detection

COPY --from=builder \
    /opt/tensorflow_models/research/slim \
    /opt/tensorflow_models/research/slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/project/app:/opt/tensorflow_models/research:/opt/tensorflow_models/research/slim

ADD image_object_detection /opt/project/app/image_object_detection

RUN mkdir -p /tmp/model/download && \
    cd /tmp/model/download && \
    curl -L -O "${MODEL_URL}" && \
    mkdir -p /tmp/model/files && \
    tar -xzvf /tmp/model/download/*.tar.gz -C /tmp/model/files && \
    mkdir -p /opt/models && \
    cp /tmp/model/files/*/frozen_inference_graph.pb /opt/models/model.pb && \
    rm -r /tmp/model; fi

WORKDIR /opt/project/app

CMD python image_object_detection/app.py
