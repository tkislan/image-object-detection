import os

from object_detection.utils import label_map_util

PATH_TO_LABELS = os.path.join('/opt/tensorflow_models/research/object_detection/data', 'mscoco_label_map.pbtxt')

__category_index = None


def category_index():
    global __category_index
    if __category_index is None:
        __category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS,
            use_display_name=True
        )

    return __category_index
