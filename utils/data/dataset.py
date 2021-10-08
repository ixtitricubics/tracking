import os
from typing import ItemsView 
import cv2
import numpy as np 
def euclidean(a,b):
    return np.sqrt(np.sum(np.square(a-b),-1))

class Dataset:
    """
    Dataset class to hold the info about registered people
    """   
    class Items:
        EMBEDDINGS="embeddings"
        IMAGES="images"
        TEMP_EMBEDDINGS="tmp_embeddings"
        TEMP_IMAGES="tmp_images"
        ASPECT_RATIO="aspect_ratio"
     
    def __init__(self, feat_extractor) -> None:
        self.feat_extractor = feat_extractor
        self.db = {}
        self.load()
        self.temporary_memory_limit = 100
            
    def register(self, imgs):
        # embeddings = []
        # calculate embeddings 
        embeddings = self.feat_extractor.run(imgs).cpu().numpy()
        aspects = []
        for i in range(len(imgs)):
            aspects.append(imgs[i].shape[:2])
        aspects = np.array(aspects)
        aspect_ratio = np.sum(aspects[:, 0])/np.sum(aspects[:,1])
        # new id
        new_id = len(self.db)
        self.db[new_id] = {
            Dataset.Items.IMAGES:imgs, 
            Dataset.Items.EMBEDDINGS:embeddings, 
            Dataset.Items.TEMP_EMBEDDINGS:[], Dataset.Items.TEMP_IMAGES:[],
            Dataset.Items.ASPECT_RATIO:aspect_ratio
            }
    
    def update(self, id, imgs):
        embeddings = list(self.feat_extractor.run(imgs).cpu().numpy())
        self.db[id][Dataset.Items.TEMP_EMBEDDINGS].extend(embeddings)
        self.db[id][Dataset.Items.IMAGES].append(imgs)
        
        if(len(self.db[id][Dataset.Items.IMAGES]) > self.temporary_memory_limit):
            del self.db[id][Dataset.Items.TEMP_EMBEDDINGS][0]
            del self.db[id][Dataset.Items.IMAGES][0]

    def load(self, folder="/home/ixti/Documents/projects/github/my_tracker/data/dataset"):

        print("loading dataset")
        identities = os.listdir(folder)

        # load images
        for i in range(len(identities)):
            img_names = os.listdir(os.path.join(folder, identities[i]))
            imgs =[]
            for j in range(len(img_names)):
                img = cv2.imread(os.path.join(folder, identities[i],img_names[j]), -1)
                imgs.append(img)
            self.register(imgs)    
        print("loading dataset finished")

    def single_distance(self, embedding):
        distances = []
        for i in self.db:
            distance = euclidean(np.expand_dims(embedding,0), 
            self.db[i][Dataset.Items.EMBEDDINGS])
            # print("distance", distance)
            distances.append(np.min(distance))
        
        print("distance", distances)  
        return distances
