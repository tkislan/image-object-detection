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

RUN git clone https://github.com/tensorflow/models.git /root/tensorflow_models && \
    (cd /root/tensorflow_models && git reset --hard b36872b66fab34fbf71d22388709de5cd81878b4)
RUN rm -rf /root/tensorflow_models/.git

RUN (cd /root/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.)

FROM python:3.5.6-slim

#ADD docker/minio.patch /docker/

RUN apt-get update && \
    apt-get install -y libtiff5 libjpeg62-turbo zlib1g libfreetype6 liblcms2-2 libwebp6 libopenjp2-7 patch && \
    pip install tensorflow==1.12.0 pillow==5.4.1 matplotlib minio && \
#    (cd /usr/local/lib/python3.5/site-packages/minio && patch -p2 < /docker/minio.patch) && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/ && \
    rm -r /root/.cache/pip

COPY --from=builder \
    /usr/local/lib/python3.5/site-packages/Pillow-5.4.1-py3.5.egg-info \
    /usr/local/lib/python3.5/site-packages/
COPY --from=builder \
    /usr/local/lib/python3.5/site-packages/PIL \
    /usr/local/lib/python3.5/site-packages/PIL

COPY --from=builder \
    /root/tensorflow_models/research/object_detection \
    /root/tensorflow_models/research/object_detection

COPY --from=builder \
    /root/tensorflow_models/research/slim \
    /root/tensorflow_models/research/slim

ENV PYTHONPATH=/root/tensorflow_models/research:/root/tensorflow_models/research/slim

ADD src/ /root/app
ADD models /root/models

ENV DETECTION_MODEL_PATH=/root/models/ssdlite_mobilenet_v2_coco.pb

WORKDIR /root/app
