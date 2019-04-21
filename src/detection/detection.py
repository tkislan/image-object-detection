from typing import List

import numpy as np
import tensorflow as tf

from object_detection.utils import ops

from detection.category_index import category_index
from detection.visualizer import visualize

DEFAULT_DETECTION_CLASSES = ["person", "car", "cat"]
DEFAULT_DETECTION_THRESHOLD = 0.7


def run_inference(sess: tf.Session, image: np.array) -> dict:
    # Get handles to input and output tensors
    operations = sess.graph.get_operations()
    all_tensor_names = {output.name for op in operations for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = sess.graph.get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def get_class_name(cls):
    return category_index()[cls]['name']


def filter_classes(output_dict, classes: List[str], threshold):
    detection_values = list(
        zip(
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores']
        )
    )

    detection_values = [
        (box, cls, score) for (box, cls, score) in detection_values
        if score > threshold and get_class_name(cls) in classes
    ]

    if len(detection_values) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes, classes, scores = [np.array(list(t)) for t in zip(*detection_values)]
    return boxes, classes, scores


def detect_objects(
        sess: tf.Session,
        image: np.array,
        threshold: float = DEFAULT_DETECTION_THRESHOLD
):
    output_dict = run_inference(sess, image)

    return filter_classes(output_dict, DEFAULT_DETECTION_CLASSES, threshold)


def detect(tf_sess, image):
    boxes, classes, scores = detect_objects(tf_sess, image)
    visualize(image, boxes, classes, scores)

    unique_classes = list(set([get_class_name(cls) for cls in classes]))

    return image, unique_classes
