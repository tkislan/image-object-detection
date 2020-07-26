import os
import sys

import tensorrt as trt
from image_object_detection.utils.engine import build_engine, save_engine
from image_object_detection.utils.trt_model import model_to_uff

TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}


def trt_model_convert(model_path: str, trt_engine_path: str, trt_precision: int, batch_size: int):
    trt_logger = trt.Logger(trt.Logger.WARNING)

    uff_model_path = os.path.join(os.path.dirname(model_path), 'frozen_inference_graph.uff')

    model_to_uff(model_path, uff_model_path)

    trt_engine = build_engine(
        uff_model_path, 
        trt_logger,
        trt_engine_datatype=TRT_PRECISION_TO_DATATYPE[trt_precision],
        batch_size=batch_size
    )
    # Save the engine to file
    save_engine(trt_engine, trt_engine_path)
    

if __name__ == "__main__":
    if len(sys.argv != 5):
        print(f'Usage: {sys.argv[0]} <model_path> <engine_path> <trt_precision - 16|32> <batch_size>')

    model_path = sys.argv[1]
    engine_path = sys.argv[2]
    trt_precision = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    trt_model_convert(model_path, engine_path, trt_precision, batch_size)
