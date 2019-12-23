import os


DETECTION_CLASSES = os.environ.get('DETECTION_CLASSES', 'person').split(',')
DETECTION_THRESHOLD = float(os.environ.get('DETECTION_THRESHOLD') or 0.5)
