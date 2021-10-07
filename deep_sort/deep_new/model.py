import torch
import torchvision.models as models
import torchvision.utils
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F



class MobileNet_Large(nn.Module):
    def __init__(self,train_fe=True,last_layer='', classes=30, use_cuda=True):
        super(MobileNet_Large, self).__init__()
        densenet = models.mobilenet_v3_large(pretrained=True)
        # remove last layer then reduce number of channel to 512
        self.feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
                                                
        self.linear = nn.Sequential(nn.Linear(960,512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(512,classes))
                                    
        if train_fe:
            # freeze parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()
            self.linear = self.linear.cuda()


    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.contiguous().view(features.size(0), -1)
        features = self.linear(features)
        preds = torch.argmax(F.softmax(features), -1)
        return preds






