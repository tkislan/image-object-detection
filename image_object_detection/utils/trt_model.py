# UFF conversion functionality

import tensorrt as trt
import graphsurgeon as gs
import uff

from image_object_detection.utils.model_data import ModelData

def ssd_unsupported_nodes_to_plugin_nodes(ssd_graph):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_inception_v2_coco_2017_11_17 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """
    # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
    channels = ModelData.get_input_channels()
    height = ModelData.get_input_height()
    width = ModelData.get_input_width()

    Input = gs.create_plugin_node(name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[1, channels, height, width])

    PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=91,
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
        "MultipleGridAnchorGenerator/Identity": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    # Create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)
    return ssd_graph

def model_to_uff(model_path, output_uff_path):
    """Takes frozen .pb graph, converts it to .uff and saves it to file.

    Args:
        model_path (str): .pb model path
        output_uff_path (str): .uff path where the UFF file will be saved
    """
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph = ssd_unsupported_nodes_to_plugin_nodes(dynamic_graph)

    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True
    )