import os
from queue import Queue

from minio import Minio
import tensorflow as tf

from detection.detection import detect
from detection.image import load_image, save_image
from minio_utils.config import BUCKET_NAME, INPUT_PREFIX, OUTPUT_PREFIX
from minio_utils.client import create_client
from detection.session import create_detection_session
from minio_utils.events import MinioEventThread, iterate_objects
from minio_utils.metadata import normalize_metadata
from utils.signal_listener import SignalListener

DETECTION_MODEL_PATH = os.environ.get('DETECTION_MODEL_PATH')

if DETECTION_MODEL_PATH is None:
    raise ValueError('DETECTION_MODEL_PATH environment variable missing')


def process_file_object(mc: Minio, tf_sess: tf.Session, bucket_name: str, key: str, output_prefix: str):
    file_name = os.path.basename(key)

    print('Processing file: {}'.format(key))

    try:
        tmp_input_file_path = os.path.join('/tmp', file_name)
        tmp_output_file_path = os.path.join('/tmp', 'output_' + file_name)

        print('Downloading file: {}'.format(key))
        ret = mc.fget_object(bucket_name, key, tmp_input_file_path)

        try:
            image_np = load_image(tmp_input_file_path)
        finally:
            os.remove(tmp_input_file_path)

        print('Detecting objects in file: {}'.format(tmp_input_file_path))
        image_np, classes = detect(tf_sess, image_np)
        print('Detected classes: {}'.format(','.join(classes)))

        input_metadata = normalize_metadata(ret.metadata)
        metadata = {
            **input_metadata,
            "x-amz-meta-classes": ','.join(classes),
        }

        try:
            save_image(image_np, tmp_output_file_path)

            output_key = os.path.join(output_prefix, file_name)

            print('Uploading file to: {}'.format(output_key))
            mc.fput_object(
                bucket_name,
                output_key,
                tmp_output_file_path,
                metadata=metadata
            )
        finally:
            os.remove(tmp_output_file_path)
    finally:
        mc.remove_object(bucket_name, key)

    print('Finished processing file')


def safe_process_file_object(mc: Minio, tf_sess: tf.Session, bucket_name: str, key: str, output_prefix: str):
    try:
        process_file_object(mc, tf_sess, bucket_name, key, output_prefix)
    except OSError as os_error:  # Ignore and log invalid images
        print(os_error)


def detection_loop(q: Queue, bucket_name: str, input_prefix: str, output_prefix: str):
    tf_sess = create_detection_session(DETECTION_MODEL_PATH)
    mc = create_client()

    objects = mc.list_objects_v2(bucket_name, prefix=input_prefix)
    for obj in objects:
        safe_process_file_object(mc, tf_sess, obj.bucket_name, obj.object_name, output_prefix)

    while True:
        event = q.get()
        if event is None:
            break

        for obj_bucket_name, obj_key in iterate_objects(event):
            safe_process_file_object(mc, tf_sess, obj_bucket_name, obj_key, output_prefix)


def main():
    q = Queue()

    SignalListener(q)

    with MinioEventThread(q, BUCKET_NAME, INPUT_PREFIX):
        detection_loop(q, BUCKET_NAME, INPUT_PREFIX, OUTPUT_PREFIX)


if __name__ == '__main__':
    main()
