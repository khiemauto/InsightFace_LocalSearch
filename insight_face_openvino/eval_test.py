import warnings
warnings.simplefilter('ignore')

import math
import os
import pickle
import tarfile
import time
import json
from numba import jit

import cv2
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

from insight_face.config import device
from insight_face.data_gen import data_transforms
from insight_face.utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes
from insight_face.models import resnet101
from insight_face.utils import parse_args

class Eval():
    def __init__(self, checkpoint):
        self.checkpoint = torch.load(checkpoint, map_location=device)
        print('loading model: {}...'.format(checkpoint))
        # self.model = self.checkpoint['model'].module
        self.args = {"pretrained": False, "use_se": True}
        self.model = resnet101(self.args)
        self.model.load_state_dict(self.checkpoint)
        self.model.to(device)
        self.model.eval()
        self.transformer = data_transforms['val']
        

    def crop(self, folder):
        if os.path.isdir(folder):
            files = os.listdir(folder)
            if not os.path.exists(folder + '_crop'):
                os.makedirs(folder + '_crop')
        else:
            raise ValueError("Folder is not exist")

        for file in files:
            filepath = os.path.join(folder, file)
            new_fn = os.path.join(folder + '_crop', file)
            bounding_boxes, landmarks = get_central_face_attributes(filepath)
            img = align_face(filepath, landmarks)
            cv2.imwrite(new_fn, img)

   
    def gen_feature(self, faces) :
        temp = torch.Tensor([])
        for face in faces:
            face = cv2.resize(face, (112,112))
            face = self.get_image(face).unsqueeze(0)

            if temp.shape[0] == 0:
                temp = face
            else:
                temp = torch.cat((temp, face),0)

        feature = self.model(temp.to(device)).cpu().detach().numpy()
        for i in range(feature.shape[0]):
            feature[i] = feature[i]/np.linalg.norm(feature[i])

        return feature

    def get_image(self, img):
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = self.transformer(img)
        return img.to(device)

    def check_same_person(self,face_feature, list_faces_embedded):
        threshold = 65.5
        result = []
        for face_embedded in list_faces_embedded :
            cosine = np.dot(face_feature, face_embedded)
            cosine = np.clip(cosine, -1.0, 1.0)
            theta = math.acos(cosine)
            theta = abs(theta * 180 / math.pi)
            #print(theta)
            if theta < threshold:
                result.append(True)
            else :
                result.append(False)

        return result

    def check_same_person_euclid(self,face_feature, list_face_embedded):
        threshold = 1.09
        result = []
        #Compare 2 feature with euclid distance
        dists = torch.cdist(face_feature.unsqueeze(0), list_face_embedded, p=2)
        #Compare distance with threshold
        dists = dists < threshold
        #Convert tensor to numpy
        dists = dists.cpu().detach().numpy()
        
        return [dists[0][i] for i in range(dists.shape[1])]
        





