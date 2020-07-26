import os

TENSORRT = os.environ.get('TENSORRT', '0') == '1'
DETECTION_CLASSES = os.environ.get('DETECTION_CLASSES', 'person,cat').split(',')
DETECTION_THRESHOLD = float(os.environ.get('DETECTION_THRESHOLD') or 0.7)
