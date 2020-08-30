import os
import time
from typing import List

import numpy as np

import image_object_detection.utils.common as common
import image_object_detection.utils.engine as engine_utils  # TRT Engine creation/save/load utils
import tensorrt as trt
from image_object_detection.utils.base_inference import BaseInferenceWrapper
from image_object_detection.utils.coco import COCO_CLASSES_LIST
from image_object_detection.utils.detection_output import DetectionOutput
from image_object_detection.utils.model_data import ModelData
from image_object_detection.utils.trt_model import model_to_uff

TRT_PRECISION = int(os.environ.get('TRT_PRECISION', 32))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def analyze_tensorrt_prediction(detection_out, pred_start_idx):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label_idx = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)

    return image_id, label_idx, confidence, xmin, ymin, xmax, ymax


class TRTInference(BaseInferenceWrapper):
    """Manages TensorRT objects for model inference."""
    def __init__(self, model_path: str):
        """Initializes TensorRT objects needed for model inference.

        Args:
            model_path (str): path where TF graph is stored
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        trt_engine_path = os.path.join(os.path.dirname(model_path), f'engine_{TRT_PRECISION}_{BATCH_SIZE}.bin')

        if not os.path.exists(trt_engine_path):
            uff_model_path = os.path.join(os.path.dirname(model_path), 'graph.uff')

            model_to_uff(model_path, uff_model_path)

            self.trt_engine = engine_utils.build_engine(
                uff_model_path, 
                TRT_LOGGER,
                trt_engine_datatype=TRT_PRECISION_TO_DATATYPE[TRT_PRECISION],
                batch_size=BATCH_SIZE
            )
            # Save the engine to file
            engine_utils.save_engine(self.trt_engine, trt_engine_path)
        else:
            print("Loading cached TensorRT engine from {}".format(trt_engine_path))
            self.trt_engine = engine_utils.load_engine(self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = engine_utils.allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, img_np: np.ndarray):
        """
        Infers model on given image.
        """

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img_np.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keep_count_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keep_count_out

    def infer_batch(self, imgs: List[np.ndarray]):
        imgs_np = np.array(imgs)
        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(imgs_np)
        if actual_batch_size > max_batch_size:
            raise ValueError("imgs_np list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs_np.ravel())

        # ...fetch model outputs...
        [detections, keep_count] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size)
        # ...and return results.
        # return detections, keep_count
        return self._convert_output(len(imgs_np), detections, keep_count)
    
    def _convert_output(self, image_count: int, detections, keep_count) -> List[List[DetectionOutput]]:
        detection_outputs = []

        prediction_fields = len(TRT_PREDICTION_LAYOUT)
        for img_idx in range(image_count):
            image_detection_outputs = []

            img_predictions_start_idx = prediction_fields * keep_count[img_idx] * img_idx
            for det in range(int(keep_count[img_idx])):
                _, label_idx, confidence, xmin, ymin, xmax, ymax = analyze_tensorrt_prediction(detections, img_predictions_start_idx + det * prediction_fields)
                label = COCO_CLASSES_LIST[label_idx]

                if not (confidence > 0.15 and label in ['person']):
                    continue

                image_detection_outputs.append(DetectionOutput(
                    label,
                    confidence,
                    np.array(xmin, ymin, xmax, ymax),
                ))
            
            detection_outputs.append(image_detection_outputs)
            
        return detection_outputs
