from minio import Minio
from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists

from .config import HOST, PORT, ACCESS_KEY, SECRET_KEY


def create_client() -> Minio:
    return Minio(
        "{0}:{1}".format(HOST, PORT),
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )


def make_bucket_if_not_exists(mc: Minio, bucket_name: str):
    # Make a bucket with the make_bucket API call.
    try:
        mc.make_bucket(bucket_name)
    except BucketAlreadyOwnedByYou:
        pass
    except BucketAlreadyExists:
        pass
