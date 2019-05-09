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

#RUN curl http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz \
#    -o /tmp/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
#RUN tar -xzf /tmp/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz -C /tmp
RUN curl http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz \
    -o /tmp/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
RUN tar -xzf /tmp/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz -C /tmp


FROM python:3.5.6-slim

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
    /root/tensorflow_models/research/object_detection \
    /root/tensorflow_models/research/object_detection

COPY --from=builder \
    /root/tensorflow_models/research/slim \
    /root/tensorflow_models/research/slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/root/tensorflow_models/research:/root/tensorflow_models/research/slim

ADD src/ /root/app
#COPY --from=builder \
#    /tmp/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb \
#    /root/models/ssd_resnet50_v1_coco.pb
COPY --from=builder \
    /tmp/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb \
    /root/models/faster_rcnn_inception_v2_coco.pb

#ENV DETECTION_MODEL_PATH=/root/models/ssd_resnet50_v1_coco.pb
ENV DETECTION_MODEL_PATH=/root/models/faster_rcnn_inception_v2_coco.pb

WORKDIR /root/app
