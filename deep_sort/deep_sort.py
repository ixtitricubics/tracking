import numpy as np
import torch

from .deep_new.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


from utils.camera import cameras
import cv2
import time
import matplotlib.pyplot as plt
__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self,db, extractor,cam_no, model_path, max_dist=0.4, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.db = db 
        self.extractor = extractor
        
        # Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 2
        
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        self.tracker = Tracker(metric,db, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.drawer = None
        self.height, self.width = None, None
        self.history = {}
        self.cam = cameras[cam_no]
    def draw(self,):
        if(self.height is not None):
            board = np.zeros((self.height, self.width,3),dtype=np.uint8)
            keys = list(self.history.keys())
            colors = [
                [181,231,214],
                [165,117,181],
                [74,134,189],
                [148,190,222],
                [206,231,123],
                [255,219,148],
                [247,117,49],
                [247,73,148],
                [255,203,90],
                [255,170,198],
                [255,000,16],
                [204,204,204],
                [0,0,128],
                [0,128,128],
                [192,192,192],
                [218,112,214],
                [127,255,212],
            ]
            for i in range(len(keys)):
                for j in range(len(self.history[keys[i]])):
                    try:
                        board = cv2.circle(board, self.history[keys[i]][j], 4, colors[keys[i]], -1)
                        
                    except:
                        print("error occured ub deeo sirt drawubg board")
                        pass
        # cv2.imshow("history", cv2.resize(board, (800,600)))

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        self.tracker.height, self.tracker.width = self.height, self.width
        
        if(len(bbox_xywh) == 0):
            return []
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences)] # if conf>self.min_confidence

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            if(track_id in self.history):
                self.history[track_id].append((int((x1+x2)/2.), int((y1 + y2)/2.)))
            else:
                self.history[track_id]= [(int((x1+x2)/2.), int((y1 + y2)/2.))]
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        # self.draw()
        return outputs
    def merge_track(self, from_id, to_id):
        self.tracker.merge_track(from_id, to_id)
    def delete_track(self, id):
        self.tracker.delete_track(id)
    def change_pos(self, track_id, new_pos):
        self.tracker.change_pos(track_id, new_pos)
    
    def get_feat(self, track_id):
        return self.tracker.get_feat(track_id)

    def add_feats(self, track_id, feats):
        self.tracker.add_feats(track_id, feats)

    def update_pose(self, poses, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        self.tracker.height, self.tracker.width = self.height, self.width
        
        if(len(poses) == 0):
            return []
        # generate detections
        features, boxes, positions, confidences_ = self._get_pose_features(poses, ori_img, confidences,self.min_confidence)
        if(features is None):
            return []
        bbox_tlwh = self._xyxy_to_tlwh_array(boxes)
        
        # print("confidences")
        
        # print(confidences)
        detections = [Detection(bbox_tlwh[i], conf, features[i], positions[i]) for i,conf in enumerate(confidences_) ]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, confidences_)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 2:
                continue
            box = track.to_tlwhxwyw()
            pos = box[4:]
            box = box[:4]
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            # if(track_id in self.history):
            #     self.history[track_id].append((int((x1+x2)/2.), int((y1 + y2)/2.)))
            # else:
            #     self.history[track_id]= [(int((x1+x2)/2.), int((y1 + y2)/2.))]
            outputs.append(np.array([x1,y1,x2,y2,*pos, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        # self.draw()
        return outputs
        


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        bbox_xyxy[:,2] = bbox_xyxy[:,2] - bbox_xyxy[:,0]
        bbox_xyxy[:,3] = bbox_xyxy[:,3] - bbox_xyxy[:,1]
        return bbox_xyxy

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features =  self.extractor(im_crops)
            features = features.cpu().numpy()
        else:
            features = np.array([])
        return features
    

    def _get_pose_features(self, poses, ori_img, confidences_, threshold=0.4):
        boxes = []
        positions = []
        crops = []
        confidences = []

        for i in range(len(poses)):
            xyxy, pos, crop = self.get_crop(ori_img, poses[i], self.cam, threshold=0.2)
            
            if(pos is None):
                pos = [-1, -1]
            if(xyxy is None or crop is None or confidences_[i] < threshold): continue

            boxes.append(xyxy)            
            positions.append(pos)
            crops.append(crop)
            confidences.append(confidences_[i])
        # print(crops)
        # print("crops", len(crops))
        if crops:
            features =  self.extractor(crops)
            features = features.cpu().numpy()
        else:
            features = None
        
        return features, np.float32(boxes), np.float32(positions), np.float32(confidences)

    def get_crop(self, img, pose, cam, threshold=0.2):
        """
        pose is a 25x3 array
        """
        # print(pose)
        if(pose == []):
            return None, None, None
        pts = np.int32(pose[:,:2])
        scores = pose[:,2]
        mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        first_pairs = [1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24]
        select = scores > threshold
        if(len(pts[select]) == 0):
            return None, None, None
        # min_x_in = np.argmin(pts[:,0])
        # min_y_in = np.argmin(pts[:,1])
        # max_x_in = np.argmax(pts[:,0])
        # max_y_in = np.argmax(pts[:,1])
        # pts[select[min_x_in], 0] -= 10 
        # pts[select[min_y_in], 1] -= 20
        # pts[select[max_x_in],0] += 10
        # pts[select[max_y_in], 1] += 20

        min_x = np.min(pts[select,0]) - 10
        min_x = max(min_x, 0)
        min_y = np.min(pts[select,1]) - 20
        min_y = max(min_y, 0)
        max_x = np.max(pts[select,0]) + 10
        max_y = np.max(pts[select,1]) + 20

        # pts for the head and shoulders
        headpts_ids = [0, 1, 2, 5,15,16,17,18]
        head_pts = []
        for i in range(len(headpts_ids)):
            if(scores[headpts_ids[i]] < threshold): continue
            head_pts.append(pts[headpts_ids[i]])
        if(len(head_pts)   > 4):
            head_pts = np.array(head_pts, dtype=np.int32)
            # print("head_pts", head_pts.shape)
            
            min_head_x = np.min(head_pts[:,0]) - 10
            min_head_x = max(min_head_x, 0)
            min_head_y = np.min(head_pts[:,1]) - 20
            min_head_y = max(min_head_y, 0)
            max_head_x = np.max(head_pts[:,0]) + 10
            max_head_y = np.max(head_pts[:,1]) + 20
            box = [min_head_x, min_head_y, max_head_x, max_head_y]
        else:
            box = None
            return None, None, None
        # height = max_y - min_y
        if(scores[21] > threshold and scores[24] > threshold):
            heel_centre = pts[21] if pts[21][1] >pts[24][1] else pts[24] #(pts[21] + pts[24])//2
        elif(scores[21] > threshold):
            heel_centre = pts[21]
        elif(scores[24] > threshold):
            heel_centre = pts[24]
        else:
            heel_centre = None
        if(heel_centre is None):
            pos = None
        else:
            pos = tuple(cam.convert([*heel_centre, 1]))
        poly_pts = []
        for i in range(0, len(first_pairs), 2):
            if(scores[first_pairs[i]] < threshold or scores[first_pairs[i+1]] < threshold): 
                    continue
            poly_pts.append(pts[first_pairs[i]])
            poly_pts.append(pts[first_pairs[i+1]])

            # cv2.line(mask, pts[first_pairs[i]], pts[first_pairs[i+1]], (255, 255, 255), 28)
        poly_pts = np.int32(poly_pts)
        cv2.polylines(mask, pts =[poly_pts], color=(255,255,255),isClosed=True, thickness=20, lineType=-1)

        # print(height)
        img = cv2.bitwise_and(img, img, mask=mask)
        # if(False):
        #     cv2.imshow("cut", img)
        #     cv2.waitKey(0)
        if(False):
            current = time.time()
            print("showing images")            
            tmp = np.copy(img[min_y:max_y, min_x:max_x])
            print(tmp.shape)
            
            # plt.imshow(tmp)
            # plt.show()
            # plt.pause(1)
            # key = input("<Hit Enter To Close>")
            # print(key, type(key))
            
            # cv2.imshow("img", tmp)
            # if cv2.waitKey(1) & 0xFF == 27:        
            #     print("smth")
            # else:
            #     path = "/home/ixti/Documents/projects/github/my_tracker/data/" + str(round(current,1)) + ".jpg"
            #     # print(path, tmp.shape)
            #     cv2.imwrite(path, tmp)
            # # cv2.destroyAllWindows()
            # if(ret == ord("s")):
            #     
            

        return box, pos, img[min_y:max_y, min_x:max_x]