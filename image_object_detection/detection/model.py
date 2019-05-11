import tensorflow as tf


def load_graph(file_path: str) -> tf.Graph:
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        return tf.import_graph_def(od_graph_def, name='')
