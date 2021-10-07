import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector_yolo4
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from models.pose import OpenPose

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

        self.vdos = {}
        self.deepsorts = {}
        for i in range(len(self.video_paths)):
            # print(i, self.video_paths[i])
            self.vdos[i]= {"cap":cv2.VideoCapture(), "width":-1, "height":-1}
            self.deepsorts[i] = build_tracker(cfg, use_cuda=use_cuda, cam_no=i)

    def __enter__(self):
        for i in range(len(self.vdos)):
            print(i, self.video_paths[i])
            self.vdos[i]["cap"].open(self.video_paths[i])
            self.vdos[i]["width"] = int(self.vdos[i]["cap"].get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vdos[i]["height"] = int(self.vdos[i]["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vdos[i]["cap"].set(1, 1200)

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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        
        while True:
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            
            start = time.time()
            rets, imgs = [],[]
            for i in range(len(self.vdos)):
                ret, ori_im = self.vdos[i]["cap"].read()
                rets.append(ret)
                imgs.append(ori_im)
            if(np.sum(rets) == 0):
                break
            elif(np.sum(rets) <len(self.video_paths)):
                continue
            poses_all, cls_conf_all = [], []
            outputs_all = []
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
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(imgs[i], bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsorts[i]._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))

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

            # if(len(outputs) > 0):
            #     # logging:
            #     self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
            #                     .format(end - start, 1 / (end - start), len(poses), len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov4.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
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
    names = ["4p-c0.avi","4p-c1.avi", "4p-c2.avi", "4p-c3.avi"]
    paths= [root + names[i] for i in range(len(names))]
    with VideoTracker(cfg, args, video_paths=paths) as vdo_trk:
        vdo_trk.run()
