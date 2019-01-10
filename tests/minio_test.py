import unittest
import io
import time
from queue import Queue

from minio_utils.client import create_client, make_bucket_if_not_exists
from minio_utils.events import MinioEventThread
from .test_utils import purge_bucket, get_random_bucket_name


class MinioBucketTest(unittest.TestCase):

    def setUp(self):
        self.mc = create_client()
        self.bucket_name = get_random_bucket_name()

    def tearDown(self):
        purge_bucket(self.mc, self.bucket_name)

    def test_make_bucket(self):
        make_bucket_if_not_exists(self.mc, self.bucket_name)

        bucket_exists = self.mc.bucket_exists(self.bucket_name)
        self.assertTrue(bucket_exists)


class MinioTest(unittest.TestCase):

    def setUp(self):
        self.mc = create_client()
        self.bucket_name = get_random_bucket_name()
        make_bucket_if_not_exists(self.mc, self.bucket_name)

    def tearDown(self):
        purge_bucket(self.mc, self.bucket_name)

    def test_minio_events(self):
        object_name = 'test/test.jpg'
        data = b'abcdef'
        metadata = {'mqtt_topic': 'sensor/camera1'}

        q = Queue()

        with MinioEventThread(q, self.bucket_name, ''):
            time.sleep(1)  # Wait for connection to establish

            self.mc.put_object(self.bucket_name, object_name, io.BytesIO(data), len(data), metadata=metadata)

            print('Getting event from queue')
            event = q.get(timeout=5)

            self.assertIsNotNone(event)
