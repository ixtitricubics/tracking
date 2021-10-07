import sys
dir_path = "/home/ixti/Documents/projects/github/openpose/build/python"
sys.path.append(dir_path);
from openpose import pyopenpose as op

class OpenPose:
    def __init__(self):       
        params = dict()
        params["model_folder"] = "/home/ixti/Documents/projects/github/openpose/models/"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
    def run(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return ([], []) if datum.poseKeypoints is None else datum.poseKeypoints, datum.poseScores
    def __call__(self,img):
        return self.run(img)