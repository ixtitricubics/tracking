import os
import cv2
import time
import argparse

from numpy.lib.ufunclike import fix
import torch
import warnings
import numpy as np

# from detector import build_detector_yolo4
from deep_sort import build_tracker, db
from utils.data.dataset import euclidean
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from models.pose import OpenPose
from deep_sort.sort.nn_matching import _cosine_distance

class VideoTracker(object):
    def __init__(self, cfg, args, video_paths):
        self.cfg = cfg
        self.args = args
        self.video_paths = video_paths
        self.video_path = video_paths[0]
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        
        if args.display:
            for i in range(len(video_paths)):
                cv2.namedWindow("img" + str(i), cv2.WINDOW_NORMAL)
                cv2.resizeWindow("img" + str(i), args.display_width, args.display_height)

        self.detector = OpenPose()

        self.vdos = []
        self.deepsorts = []
        for i in range(len(self.video_paths)):
            # print(i, self.video_paths[i])
            self.vdos.append({"cap":cv2.VideoCapture(), "width":-1, "height":-1})
            self.deepsorts.append(build_tracker(cfg, use_cuda=use_cuda, cam_no=i))

    def __enter__(self):
        for i in range(len(self.vdos)):
            self.vdos[i]["cap"].open(self.video_paths[i])
            self.vdos[i]["width"] = int(self.vdos[i]["cap"].get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vdos[i]["height"] = int(self.vdos[i]["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vdos[i]["cap"].set(1, 300)

        # if self.args.save_path:
        #     os.makedirs(self.args.save_path, exist_ok=True)

        #     # path of saved video and results
        #     self.save_video_path = os.path.join(self.args.save_path, "results.avi")
        #     self.save_results_path = os.path.join(self.args.save_path, "results.txt")

        #     # create video writer
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #     self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

        #     # logging
        #     self.logger.info("Save results to {}".format(self.args.save_path))

        return self
    def draw(self, outputs_all):
        board = np.zeros((1000, 700,3),dtype=np.uint8)
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
        # for each camera
        for i in range(len(outputs_all)):
            # for each track
            for j in range(len(outputs_all[i])):
                x,y,w,h,xw,yw,track_id = outputs_all[i][j]
                try:
                    board = cv2.circle(board, (xw, yw), 4, colors[i], -1)
                    cv2.putText(board,  str(i) + "_" + str(track_id), (xw+2, yw-2), 0, 0.5, 255)
                except Exception as e:
                    print(e) 
                    # print("error occured ub deeo sirt drawubg board")
                    pass
        cv2.imshow("map", board)
    def draw_(self, result_positions):
        board = np.zeros((1000, 700,3),dtype=np.uint8)
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
        for person_id in result_positions:
            xw,yw = result_positions[person_id]
            try:
                board = cv2.circle(board, (int(xw), int(yw)), 4, colors[person_id], -1)
                cv2.putText(board,  str(person_id), (xw+2, yw-2), 0, 0.5, 255)
            except Exception as e:
                print(e) 
                # print("error occured ub deeo sirt drawubg board")
                pass
        cv2.imshow("map_mean", board)
                    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        
        while True:
            idx_frame += 1
            for i in range(len(self.vdos)):
                    _ = self.vdos[i]["cap"].grab()
            if idx_frame % self.args.frame_interval:
                print(idx_frame)
                continue
            
            start = time.time()
            rets, imgs = [],[]
            for i in range(len(self.vdos)):
                ret, ori_im = self.vdos[i]["cap"].retrieve()
                rets.append(ret)
                imgs.append(ori_im)
            if(np.sum(rets) == 0):
                break
            elif(np.sum(rets) <len(self.video_paths)):
                continue
            poses_all, cls_conf_all = [], []
            outputs_all = []
            positions = {}

            for i in range(len(self.vdos)):
                poses, cls_conf = self.detector(imgs[i])
                outputs = self.deepsorts[i].update_pose(poses, cls_conf,imgs[i])
                poses_all.append(poses)
                cls_conf_all.append(cls_conf)
                outputs_all.append(outputs)
                

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    world_coords = outputs[:, 4:6]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(imgs[i], bbox_xyxy, identities)
                    for j in range(len(identities)):
                        positions.setdefault(identities[j], []).append([world_coords[j], i])
                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsorts[i]._xyxy_to_tlwh(bb_xyxy))
                    results.append((idx_frame - 1, bbox_tlwh, identities))
            
            # mean means )))))))))))))))))))))))))))))))))

            result_positions = {}
            fixes = {}            
            for person_id in positions:
                if(person_id < len(db.db)+1):
                    mean_position = np.mean(positions[person_id][0], 0)
                    result_positions[person_id] = mean_position
                else:
                    for j in range(len(positions[person_id])):
                        mean_position, cam_id = positions[person_id][j]
                        fixes[(person_id, cam_id)] = mean_position
            
            
            # if there is data in fixes fix it )))
            if(len(result_positions) > 0):
                for fix_id in fixes:
                    print(fixes)
                    print("starting to fix", fix_id)
                    losses = {}            
                    for target_id in result_positions:
                        losses[target_id]= euclidean(fixes[fix_id], result_positions[target_id])
                    
                    corr_id = min(losses, key=losses.get)
                    
                    if(losses[corr_id]< 60): # 50 sm
                        feat_corr = self.deepsorts[cam_ind].get_feat(corr_id)
                        feat_fix = self.deepsorts[cam_ind].get_feat(fix_id[0])
                        dist = _cosine_distance([feat_corr], [feat_fix])[0][0]
                        print("corrected ", fix_id, " _to_", corr_id, " loss=", losses[corr_id], dist)    
                        if(dist < 0.32):
                            self.deepsorts[fix_id[1]].merge_track(fix_id[0], corr_id)
                    else:
                        # pass
                        self.deepsorts[fix_id[1]].delete_track(fix_id[0])
                    # changes.append(np.agrmin(losses)
                    # changes.append([fix_id, corr_id])
            
            # include latest features to all cameras
            for target_id in result_positions:
                new_feats = []
                for cam_ind in range(len(self.deepsorts)):
                    new_feats.append(self.deepsorts[cam_ind].get_feat(target_id))                
                for cam_ind in range(len(self.deepsorts)):
                    self.deepsorts[cam_ind].add_feats(target_id, new_feats)

            # change the newly generated to id to the fixed one 
            
            # print(np.shape(outputs_all))
            self.draw(outputs_all)
            self.draw_(result_positions)
            end = time.time()

            # if self.args.display:
            for i in range(len(imgs)):
                cv2.imshow("img" + str(i), cv2.resize(imgs[i], (self.cfg.display_width, self.cfg.display_height)))
            
            # save 
            # self.writer.write(ori_im)
            # # save results
            # # write_results(self.save_results_path, results, 'mot')
            
            ret= cv2.waitKey(3)
            if(ret == 27):
                break

            if(len(result_positions) > 0):
            #     # logging:
                self.logger.info("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov4.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--display_width", type=int, default=400)
    parser.add_argument("--display_height", type=int, default=300)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    cfg.display_width = args.display_width
    cfg.display_height = args.display_height
    root = "/home/ixti/Documents/projects/github/my_tracker/data/train/lab/"
    names = ["4p-c0.avi","4p-c1.avi" , "4p-c2.avi" ,  "4p-c3.avi" ]#  , 
    paths= [root + names[i] for i in range(len(names))]
    with VideoTracker(cfg, args, video_paths=paths) as vdo_trk:
        vdo_trk.run()
