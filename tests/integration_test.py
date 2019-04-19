import tempfile
import threading
import time
import unittest
import os
from queue import Queue

import minio

from app import DETECTION_MODEL_PATH, process_file_object, detection_loop
from detection.image import load_image
from detection.session import create_detection_session
from minio_utils.client import create_client, make_bucket_if_not_exists
from minio_utils.events import MinioEventThread, iterate_objects
from minio_utils.metadata import normalize_metadata
from tests.test_utils import get_random_bucket_name, purge_bucket

DATA_IMG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img')
IMAGE_PATH = os.path.join(DATA_IMG_DIR, 'street.jpg')
INVALID_IMAGE_PATH = os.path.join(DATA_IMG_DIR, 'invalid.txt')


class TemporaryFileName:
    def __enter__(self):
        fd, self.__file_path = tempfile.mkstemp(suffix='jpg')
        os.close(fd)
        return self.__file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.__file_path)


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.mc = create_client()
        self.bucket_name = get_random_bucket_name()
        self.input_prefix = 'test/'
        self.output_prefix = 'output_test/'
        self.image_name = 'test.jpg'
        self.object_name = os.path.join(self.input_prefix, self.image_name)
        self.output_object_name = os.path.join(self.output_prefix, self.image_name)
        self.metadata = {'mqtt_topic': 'sensor/camera1'}

        make_bucket_if_not_exists(self.mc, self.bucket_name)

        self.tf_sess = create_detection_session(DETECTION_MODEL_PATH)

    def tearDown(self):
        purge_bucket(self.mc, self.bucket_name)

    def check_image_output(self):
        with TemporaryFileName() as tmpfile:
            ret = self.mc.fget_object(self.bucket_name, self.output_object_name, tmpfile)
            tmp_image_metadata = normalize_metadata(ret.metadata)

            tmp_image = load_image(tmpfile)

            self.assertEqual((480, 640, 3), tmp_image.shape)
            self.assertListEqual(sorted(list(self.metadata.keys()) + ['classes']), sorted(tmp_image_metadata.keys()))
            self.assertEqual(self.metadata['mqtt_topic'], tmp_image_metadata['mqtt_topic'])
            self.assertListEqual(['car', 'person'], sorted(tmp_image_metadata['classes'].split(',')))

    def wait_check_image_output(self, timeout=30):
        i = 0
        while True:
            try:
                self.check_image_output()
                break
            except minio.error.NoSuchKey as e:
                if i >= timeout:
                    raise e
            time.sleep(1)
            i += 1

    def test_valid_image(self):
        q = Queue()

        with MinioEventThread(q, self.bucket_name, self.input_prefix):
            time.sleep(1)  # Wait for connection to establish

            self.mc.fput_object(self.bucket_name, self.object_name, IMAGE_PATH, metadata=self.metadata)

            event = q.get(timeout=5)

        self.assertIsNotNone(event)

        bucket_name, key = next(iterate_objects(event))
        process_file_object(self.mc, self.tf_sess, bucket_name, key, self.output_prefix)

        self.check_image_output()

    def test_invalid_image(self):
        q = Queue()

        with MinioEventThread(q, self.bucket_name, self.input_prefix):
            time.sleep(1)  # Wait for connection to establish

            self.mc.fput_object(self.bucket_name, self.object_name, INVALID_IMAGE_PATH, metadata=self.metadata)

            event = q.get(timeout=5)

        self.assertIsNotNone(event)

        bucket_name, key = next(iterate_objects(event))

        with self.assertRaises(OSError):
            process_file_object(self.mc, self.tf_sess, bucket_name, key, self.output_prefix)

    def test_app_listener(self):
        q = Queue()

        with AppThread(q, self.bucket_name, self.input_prefix, self.output_prefix):
            time.sleep(5)  # Wait TF Session to load, and for connection to establish

            self.mc.fput_object(self.bucket_name, self.object_name, IMAGE_PATH, metadata=self.metadata)

            self.wait_check_image_output()

            q.put(None)


class AppThread(threading.Thread):
    def __init__(self, q: Queue, bucket_name: str, input_prefix: str, output_prefix: str):
        super().__init__()
        self.__q = q
        self.__bucket_name = bucket_name
        self.__input_prefix = input_prefix
        self.__output_prefix = output_prefix

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__q.put(None)
        self.join()

    def run(self):
        with MinioEventThread(self.__q, self.__bucket_name, self.__input_prefix):
            detection_loop(self.__q, self.__bucket_name, self.__input_prefix, self.__output_prefix)
