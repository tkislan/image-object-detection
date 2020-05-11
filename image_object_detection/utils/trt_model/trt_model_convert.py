import sys

import tensorflow.contrib.tensorrt as trt

from image_object_detection.utils.trt_model.detection import build_detection_graph

def run():
    # config_path, checkpoint_path = download_detection_model(sys.argv[1], 'data')

    frozen_graph, input_names, output_names = build_detection_graph(
        config=sys.argv[1],
        checkpoint=sys.argv[2],
        score_threshold=0.3,
        iou_threshold=0.5,
        batch_size=1
    )

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )

    with open('./trt_graph.pb', 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__ == "__main__":
    run()
    print('Done')