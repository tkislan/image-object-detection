import tensorflow as tf

from .model import load_graph


def create_detection_session(model_file_path: str) -> tf.Session:
    detection_graph = load_graph(model_file_path)
    return tf.Session(graph=detection_graph)
