import cv2
import numpy as np
HG_C0 = [[0.176138,  0.647589, -63.412272],
        [-0.180912, 0.622446, -0.125533],
        [-0.000002, 0.001756,  0.102316]]
HG_C1 = [[0.177291, 0.004724, 31.224545],
         [0.169895, 0.661935, -79.781865],
         [-0.000028, 0.001888, 0.054634]]
HG_C2 = [[-0.118791, 0.077787, 64.819189],
         [0.133127, 0.069884, 15.832922],
         [-0.000001, 0.002045, -0.057759]]
HG_C3 = [[-0.142865, 0.553150, -17.395045],
         [-0.125726, 0.039770, 75.937144],
         [-0.000011, 0.001780, 0.015675]]

class Camera:
    def __init__(self,id,  H = None)->None:
        self.id = id
        self.H = np.array(H)
    def convert(self, point):
        """
        point -- pixel point in the image
        """
        pt = np.dot(self.H, point)
        res = (pt/pt[-1])[:2]
        res = np.int32(res)
        # print("convert", res)
        return res
    def convertBox2Pos(self, bbox):
        point = [bbox[0], bbox[-1], 1] 
        res = self.convert(point)
        # 10 sm is ok
        # discretize
        return np.int32(res/10)* 10

    @staticmethod
    def draw_map(positions, colors, width =1000, height=1000):
        map = np.zeros((height, width, 3), dtype=np.uint8)
        for cam_id in range(len(positions)):
            color = colors[cam_id]
            for det_id in range(len(positions[cam_id])):
                
                det_pos = np.int32(positions[cam_id][det_id]) 

                # print("det_pos", det_pos)
                cv2.putText(map,str(cam_id), (det_pos[0], det_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                cv2.rectangle(
                            map, 
                            (det_pos[0], det_pos[1]), 
                            (det_pos[0]+5, det_pos[1]+5), 
                            (int(color[0]), int(color[1]), int(color[2])),
                            2)
        return map

def draw_map_tracks(tracks, colors, width =1000, height=700):
    map = np.zeros((height, width, 3), dtype=np.uint8)
    # loop through the cameras
    for i in range(len(tracks)):
        # loop through tracks:
        for j in range(len(tracks[i])):
            color = colors[tracks[i][j].id]
            # print(color, tracks[i][j].id)
            # loop through the boxes
            for k in range(len(tracks[i][j].boxes)):
                # import pdb; pdb.set_trace()
                # print("tracks[i][j].positions[k]", tracks[i][j].positions[k])
                if(tracks[i][j].positions[k] is None):continue
                
                det_pos = np.int32(tracks[i][j].positions[k])
                # print(det_pos, str(tracks[i][j].id) + "|" + str(i), i,j,k, tracks[i][j])
                cv2.putText(map,str(tracks[i][j].id) + "|" + str(i), (det_pos[0], det_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
                cv2.rectangle(
                            map, 
                            (det_pos[0], det_pos[1]), 
                            (det_pos[0]+2, det_pos[1]+2), 
                            (int(color[0]), int(color[1]), int(color[2])),
                            2)
    return map.copy()

    


cameras = []
cameras.append(Camera(0, HG_C0))
cameras.append(Camera(1, HG_C1))
cameras.append(Camera(2, HG_C2))
cameras.append(Camera(3, HG_C3))

