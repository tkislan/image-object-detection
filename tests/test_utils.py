# import random
# import string


# def get_random_bucket_name():
#     return ''.join(random.choice(string.ascii_lowercase) for _ in range(10))


# def purge_bucket(mc, bucket_name: str):
#     for obj in mc.list_objects_v2(bucket_name, recursive=True):
#         mc.remove_object(obj.bucket_name, obj.object_name)
#     mc.remove_bucket(bucket_name)
