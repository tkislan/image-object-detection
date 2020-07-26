import os
from queue import Queue

from minio import Minio
from minio.error import NoSuchKey
import tensorflow as tf

from image_object_detection.detection.detection import detect
from image_object_detection.detection.image import load_image, save_image
from image_object_detection.detection.session import create_detection_session
from image_object_detection.minio_utils.config import BUCKET_NAME, INPUT_PREFIX, OUTPUT_PREFIX, TRAINING_PREFIX, \
    STORE_TRAINING_DATA
from image_object_detection.minio_utils.client import create_client
from image_object_detection.minio_utils.events import MinioEventThread, iterate_objects
from image_object_detection.utils.signal_listener import SignalListener

DETECTION_MODEL_PATH = '/opt/models/graph.pb'


def process_file_object(
    mc: Minio,
    tf_sess: tf.Session,
    bucket_name: str,
    key: str,
    input_prefix: str,
    output_prefix: str,
    training_prefix: str,
    store_training_data: bool
):
    file_name = os.path.basename(key)

    print('Processing file: {}'.format(key))

    tmp_input_file_path = os.path.join('/tmp', file_name)

    try:
        tmp_output_file_path = os.path.join('/tmp', 'output_' + file_name)

        print('Downloading file: {}'.format(key))
        try:
            ret = mc.fget_object(bucket_name, key, tmp_input_file_path)
        except NoSuchKey:
            print('File not found, ignoring')
            return

        print('Loading image into memory')
        raw_image_np = load_image(tmp_input_file_path)

        print('Detecting objects in file: {}'.format(tmp_input_file_path))
        image_np, classes = detect(tf_sess, raw_image_np)
        print('Detected classes: {}'.format(','.join(classes)))

        if not classes:
            return

        metadata = {
            **(ret.metadata or {}),
            "x-amz-meta-classes": ','.join(classes),
        }

        try:
            save_image(image_np, tmp_output_file_path)

            output_key = key.replace(input_prefix, output_prefix, 1)
            print('Uploading file to: {}'.format(output_key))
            mc.fput_object(
                bucket_name,
                output_key,
                tmp_output_file_path,
                metadata=metadata
            )

            if store_training_data:
                training_output_key = key.replace(input_prefix, training_prefix, 1)
                print('Uploading training file to: {}'.format(training_output_key))
                mc.fput_object(
                    bucket_name,
                    training_output_key,
                    tmp_input_file_path,
                    metadata=metadata
                )
        finally:
            os.remove(tmp_output_file_path)
    finally:
        os.remove(tmp_input_file_path)
        mc.remove_object(bucket_name, key)

    print('Finished processing file')


def safe_process_file_object(
    mc: Minio,
    tf_sess: tf.Session,
    bucket_name: str,
    key: str,
    input_prefix: str,
    output_prefix: str,
    training_prefix: str,
    store_training_data: bool
):
    try:
        process_file_object(
            mc,
            tf_sess,
            bucket_name,
            key,
            input_prefix,
            output_prefix,
            training_prefix,
            store_training_data
        )
    except OSError as os_error:  # Ignore and log invalid images
        print(os_error)


def detection_loop(
    q: Queue,
    bucket_name: str,
    input_prefix: str,
    output_prefix: str,
    training_prefix: str,
    store_training_data: bool
):
    tf_sess = create_detection_session(DETECTION_MODEL_PATH)
    mc = create_client()

    objects = mc.list_objects_v2(bucket_name, prefix=input_prefix, recursive=True)
    for obj in objects:
        if obj.is_dir:
            continue
        safe_process_file_object(
            mc,
            tf_sess,
            obj.bucket_name,
            obj.object_name,
            input_prefix,
            output_prefix,
            training_prefix,
            store_training_data
        )

    while True:
        event = q.get()
        if event is None:
            break

        for obj_bucket_name, obj_key in iterate_objects(event):
            safe_process_file_object(
                mc,
                tf_sess,
                obj_bucket_name,
                obj_key,
                input_prefix,
                output_prefix,
                training_prefix,
                store_training_data
            )


def main():
    q = Queue()

    SignalListener(q)

    with MinioEventThread(q, BUCKET_NAME, INPUT_PREFIX):
        detection_loop(q, BUCKET_NAME, INPUT_PREFIX, OUTPUT_PREFIX, TRAINING_PREFIX, STORE_TRAINING_DATA)


if __name__ == '__main__':
    main()
