import collections
import json
import threading
from queue import Queue
from typing import Iterator
from urllib.parse import unquote

from image_object_detection.minio_utils.client import create_client


def get_minio_notification_response(mc, bucket_name: str, prefix: str):
    query = {
        'prefix': prefix,
        'suffix': '.*',
        'events': ['s3:ObjectCreated:*'],
    }
    # noinspection PyProtectedMember
    return mc._url_open('GET', bucket_name=bucket_name, query=query, preload_content=False)


class MinioEventStreamIterator(collections.Iterable):
    def __iter__(self) -> Iterator:
        return self

    def __init__(self, response):
        self.__response = response
        self.__stream = response.stream()

    def __next__(self):
        while True:
            line = next(self.__stream)
            if line.strip():
                event = json.loads(line.decode('utf-8'))
                if event['Records'] is not None:
                    return event

    def close(self):
        self.__response.close()


class MinioEventThread(threading.Thread):
    def __init__(self, q: Queue, bucket_name: str, prefix: str):
        super().__init__()
        self.__q = q
        self.__bucket_name = bucket_name
        self.__prefix = prefix
        self.__event_stream_it = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        print('Running MinioEventThread')
        mc = create_client()

        while True:
            print('Connecting to minio event stream')
            response = get_minio_notification_response(mc, self.__bucket_name, self.__prefix)
            self.__event_stream_it = MinioEventStreamIterator(response)

            try:
                for event in self.__event_stream_it:
                    self.__q.put(event)
            except json.JSONDecodeError:
                response.close()
            except AttributeError:
                break

    def stop(self):
        print('Stopping event thread')

        if self.__event_stream_it is not None:
            self.__event_stream_it.close()
            self.__event_stream_it = None

        print('Joining event thread')
        self.join()
        print('Event thread joined')


def iterate_objects(event):
    records = event.get('Records', [])

    for record in records:
        bucket_name = record.get('s3', {}).get('bucket', {}).get('name')
        key = record.get('s3', {}).get('object', {}).get('key')

        if not bucket_name or not key:
            print('Invalid bucket_name and/or key', bucket_name, key)
            continue

        key = unquote(key)

        yield bucket_name, key
