import tensorflow as tf

from .model import load_graph


def create_detection_session(model_file_path: str) -> tf.Session:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.67
    detection_graph = load_graph(model_file_path)
    return tf.Session(graph=detection_graph, config=config)
