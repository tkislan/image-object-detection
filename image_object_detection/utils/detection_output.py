import numpy as np

class DetectionOutput:
    def __init__(self, label: str, confidence: float, box: np.ndarray):
        self.label = label
        self.confidence = confidence
        self.box = box