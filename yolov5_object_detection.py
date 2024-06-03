# yolov5_object_detection.py
import torch
import os
import sys

class YOLOv5:
    def __init__(self, model_name='yolov5s'):
        yolo_dir = r'C:\Users\kirankumar\slam_project\yolov5'
        if not os.path.exists(yolo_dir):
            raise FileNotFoundError(f"{yolo_dir} directory not found. Ensure YOLOv5 repository is cloned.")
        sys.path.insert(0, yolo_dir)

        from yolov5.models.common import DetectMultiBackend

        self.model = torch.hub.load(yolo_dir, model_name, source='local', pretrained=True)

    def detect(self, image):
        results = self.model(image)
        return results

def detect_objects(image):
    detector = YOLOv5()
    results = detector.detect(image)
    return results.pandas().xyxy[0].to_dict(orient="records")
