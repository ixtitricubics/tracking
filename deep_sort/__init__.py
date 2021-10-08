from .deep_sort import DeepSort
from utils.data.dataset import Dataset
from models.reid import ReidFeatExtractor

__all__ = ['DeepSort', 'build_tracker','db']
extractor = ReidFeatExtractor()
db = Dataset(extractor)

def build_tracker(cfg, use_cuda, cam_no):
    return DeepSort(db, extractor,cam_no, cfg.DEEPSORT.REID_CKPT, 
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    









