from object_detection.utils import visualization_utils

from .category_index import category_index


def visualize(image_np, boxes, classes, scores):
    # Visualization of the results of a detection.
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index(),
        use_normalized_coordinates=True,
        line_thickness=8
    )
