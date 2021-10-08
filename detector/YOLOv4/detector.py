try:
    from detector.YOLOv4 import darknet
except:
    import darknet
    import os
    # os.path.append("../../../")
import cv2
import numpy as np


def convert2relative(bbox, darknet_height, darknet_width):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height



def convert2original(orig_size, bbox, darknet_height, darknet_width):
    x, y, w, h = convert2relative(bbox, darknet_height, darknet_width)

    image_h, image_w, __ = orig_size

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

class YOLOv4(object):
    def __init__(self, cfgfile, weightfile, datafile, score_thresh=0.5, nms_thresh = 0.4):
        
        self.network, self.class_names, self.class_colors = darknet.load_network(
            cfgfile,
            datafile,
            weightfile,
            batch_size=1
        )
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)

        # constants
        self.size = self.darknet_width, self.darknet_height
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, img_for_detect, orig_shape=None, det_resized=False):
        detections = darknet.detect_image( self.network, self.class_names, img_for_detect, thresh=self.score_thresh,nms=self.nms_thresh)
        darknet.free_image(img_for_detect)
        if(det_resized):
            bboxes =  [convert2original(orig_shape, bbox, self.darknet_height, self.darknet_width) for label, conf, bbox in detections if label == "person"]
            probs =  [float(conf) for label, conf, bbox in detections if label == "person"]
            # [print(label) for label, conf, bbox in detections]
            return bboxes, probs
        return detections
    def draw(self, frame, detections):
        if frame is not None:
            detections_adjusted = []
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame.shape, bbox, self.darknet_height, self.darknet_width)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, self.class_colors)
            return image
        else:
            return frame
    def prepare_img(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.darknet_width, self.darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        return img_for_detect
def demo():
    cfg_file ="/home/ixti/Documents/projects/github/deep_sort_pytorch/detector/YOLOv4/yolov4.cfg"
    weight_file = "/home/ixti/Documents/projects/github/deep_sort_pytorch/detector/YOLOv4/yolov4_best.weights"
    datafile = "/home/ixti/Documents/projects/github/deep_sort_pytorch/detector/YOLOv4/drink_ix2.data"
    yolo = YOLOv4(cfg_file, weight_file, datafile)
    cap = cv2.VideoCapture("/home/ixti/Documents/projects/datasets/test_temp/output2.mp4")
    while(True):
        ret,img = cap.read()
        if(ret):
            img_for_detect = yolo.prepare_img(img)
            dets = yolo(img_for_detect)
            img = yolo.draw(img, dets)
        cv2.imshow("img", img)
        ret = cv2.waitKey(10)
        if(ret == 27):
            break
    cv2.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    demo()
