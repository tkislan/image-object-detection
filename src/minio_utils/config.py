import os

HOST = os.environ.get('MINIO_HOST')
PORT = int(os.environ.get('MINIO_PORT') or 9000)
ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
BUCKET_NAME = os.environ.get('MINIO_BUCKET_NAME')
INPUT_PREFIX = os.environ.get('MINIO_INPUT_PREFIX') or 'input/'
OUTPUT_PREFIX = os.environ.get('MINIO_OUTPUT_PREFIX') or 'output/'

if not HOST:
    raise ValueError('MINIO_HOST environment variable missing')

if not ACCESS_KEY or not SECRET_KEY:
    raise ValueError('MINIO_ACCESS_KEY and/or MINIO_SECRET_KEY environment variables missing')

if not BUCKET_NAME:
    raise ValueError('MINIO_BUCKET_NAME environment variable missing')

if not INPUT_PREFIX or not OUTPUT_PREFIX:
    raise ValueError('MINIO_INPUT_PREFIX and/or MINIO_OUTPUT_PREFIX environment variables missing')

if INPUT_PREFIX[-1:] != '/' or OUTPUT_PREFIX[-1:] != '/':
    raise ValueError('Invalid MINIO_INPUT_PREFIX and/or MINIO_OUTPUT_PREFIX environment variables')
