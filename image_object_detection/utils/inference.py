# Utility functions for performing image inference

from image_object_detection.utils.base_inference import BaseInferenceWrapper
import os
from typing import List

import tensorflow as tf
import numpy as np

from image_object_detection.utils.detection_output import DetectionOutput
from image_object_detection.utils.coco import COCO_CLASSES_LIST


# This class is similar as TRTInference inference, but it manages Tensorflow
class TensorflowInference(BaseInferenceWrapper):
    def __init__(self, pb_model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)

    def infer(self, img_np: np.ndarray):
        return self._run_tensorflow_graph(np.expand_dims(img_np, axis=0))

    def infer_batch(self, imgs_np: List[np.ndarray]):
        return self._run_tensorflow_graph(np.array(imgs_np))

    def _run_tensorflow_graph(self, image_input):
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes',
            'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(
                    tensor_name)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: image_input})

        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = output_dict['num_detections'].astype(np.int32)
        output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.uint8)

        return self._convert_output(output_dict)
    
    def _convert_output(self, output_dict) -> List[List[DetectionOutput]]:
        detection_outputs = []
        for image_idx, num_detections in enumerate(output_dict['num_detections']):
            image_detection_outputs = []

            for detection_idx in range(num_detections):
                label = COCO_CLASSES_LIST[output_dict['detection_classes'][image_idx][detection_idx]]
                confidence = output_dict['detection_scores'][image_idx][detection_idx]

                # if not (confidence > 0.5 and label in ['person']):
                # if not (confidence > 0.1):
                if not (confidence > 0.15 and label in ['person']):
                    continue

                image_detection_outputs.append(DetectionOutput(
                    label,
                    confidence,
                    output_dict['detection_boxes'][image_idx][detection_idx],
                ))
            
            detection_outputs.append(image_detection_outputs)
            
        return detection_outputs

