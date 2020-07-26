from typing import List

import numpy as np
import tensorflow as tf

from object_detection.utils import ops

from image_object_detection.detection.category_index import category_index
# from image_object_detection.detection.visualizer import visualize
from image_object_detection.detection.config import DETECTION_CLASSES, DETECTION_THRESHOLD
from image_object_detection.detection.detection_result import DetectionResult


def run_inference(sess: tf.Session, image: np.array) -> dict:
    # Get handles to input and output tensors
    operations = sess.graph.get_operations()
    all_tensor_names = {output.name for op in operations for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = sess.graph.get_tensor_by_name(tensor_name)
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


def get_class_name(class_id):
    return category_index()[class_id]['name']


def filter_classes(output_dict, classes: List[str], threshold):
    detection_values = list(
        zip(
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores']
        )
    )

    detection_values = [
        (box, class_id, score) for (box, class_id, score) in detection_values
        if score >= threshold and get_class_name(class_id) in classes
    ]

    if len(detection_values) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes, classes, scores = [np.array(list(t)) for t in zip(*detection_values)]
    return boxes, classes, scores


def detect_objects(
        sess: tf.Session,
        image: np.array,
        threshold: float = DETECTION_THRESHOLD
):
    output_dict = run_inference(sess, image)

    return filter_classes(output_dict, DETECTION_CLASSES, threshold)


def detect(tf_sess, image):
    boxes, classes, scores = detect_objects(tf_sess, image)
    # visualize(image, boxes, classes, scores)

    unique_classes = list(set([get_class_name(class_id) for class_id in classes]))

    return image, unique_classes
