import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
try:
    from .model import *
except:
    from model import *

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        num_classes = 35
        self.net = MobileNet_Large(train_fe=False,classes=num_classes)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict =  torch.load(model_path)
        self.net.load_state_dict(state_dict['state_dict'])
        
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (256, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.net.eval()
        self.linear_output = None
        self.net.linear[0].register_forward_hook(self.linear_hook)

    def linear_hook(self, module, input_, output):
        self.linear_output = output

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            # print("inside resize", im.shape, size, len(im_crops))
            return cv2.resize(im.astype(np.float32), size)
        # im_batch = torch.cat([self.norm(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).unsqueeze(0) for im in im_crops], dim=0).float()
        # return im_batch
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            preds = self.net(im_batch)
        return self.linear_output.cpu().numpy(), preds.cpu().numpy()


if(__name__ == '__main__'):
    img = cv2.imread("/home/ixti/Pictures/Screenshot from 2021-08-11 10-15-24.png")
    extr = Extractor("/home/ixti/Documents/projects/github/deep_sort_pytorch/deep_sort/deep_new/model_best_35classes.pth")
    print(img.shape)
    imgs = []
    imgs.append(img)
    imgs.append(img)
    imgs.append(img)
    imgs.append(img)
    imgs.append(img)
    feature, preds = extr(np.float32(imgs))
    print(preds)

