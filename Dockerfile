FROM python:3.5.6 as builder

RUN pip install --no-binary pillow pillow

RUN git clone https://github.com/tensorflow/models.git /root/tensorflow_models && \
    (cd $HOME/tensorflow_models && git reset --hard 2c4dc0c00aba5dde21923dd4f8aa1584e8ec67af)
RUN rm -rf /root/tensorflow_models/.git

RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-python-3.6.1.tar.gz \
        -o /tmp/protobuf-python.tar.gz

RUN tar -xzf /tmp/protobuf-python.tar.gz -C /tmp

RUN (cd /tmp/protobuf-3.6.1 && ./autogen.sh && ./configure && make && make install)
RUN ldconfig
#RUN (cd /tmp/protobuf-3.6.1 && ./autogen.sh && ./configure && make -C src protoc && find . -name protoc && which exit)
RUN rm -r /tmp/protobuf-3.6.1

RUN (cd /root/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.)

FROM python:3.5.6-slim

ADD docker/minio.patch /docker/

RUN apt-get update && \
    apt-get install -y libtiff5 libjpeg62-turbo zlib1g libfreetype6 liblcms2-2 libwebp6 libopenjp2-7 patch && \
    pip install tensorflow==1.12.0 pillow matplotlib minio && \
    (cd /usr/local/lib/python3.5/site-packages/minio && patch -p2 < /docker/minio.patch) && \
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
