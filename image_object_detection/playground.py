import time

from detection.detection import detect
from detection.image import load_image, save_image
from detection.session import create_detection_session


def run_detect(tf_sess, image_np):
    image_np, classes = detect(tf_sess, image_np)


def run_model(model_path, input_file_path, output_file_path):
    tf_sess = create_detection_session(model_path)
    image_np = load_image(input_file_path)

    start = time.process_time()
    for _ in range(30):
        run_detect(tf_sess, image_np)
    end = time.process_time()
    print(model_path + ' took ' + str(end - start) + ' to finish')


def run():
    input_file_path = './__playground/img/front_camera_1556079678.jpg'
    output_file_path = './output.jpg'

    run_model('/opt/models/faster_rcnn_resnet101_kitti.pb', input_file_path, output_file_path)
    run_model('/opt/models/faster_rcnn_inception_v2_coco.pb', input_file_path, output_file_path)
    run_model('/opt/models/ssdlite_mobilenet_v2_coco.pb', input_file_path, output_file_path)
    run_model('/opt/models/ssd_resnet50_v1.pb', input_file_path, output_file_path)


if __name__ == '__main__':
    run()
