from .YOLOv3 import YOLOv3
from.YOLOv4 import YOLOv4 

__all__ = ['build_detector_yolo', 'build_detector_yolo4']

def build_detector_yolo(cfg, use_cuda):
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)

def build_detector_yolo4(cfg, use_cuda):
    return YOLOv4(cfg.YOLOV4.CFG, cfg.YOLOV4.WEIGHT, cfg.YOLOV4.DATA_FILE, 
                    score_thresh=cfg.YOLOV4.SCORE_THRESH, nms_thresh=cfg.YOLOV4.NMS_THRESH)
