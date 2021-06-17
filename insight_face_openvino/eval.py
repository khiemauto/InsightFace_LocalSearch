import warnings
warnings.simplefilter('ignore')

import math
import os
import pickle
import tarfile
import time
import json

import cv2
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes
from models import resnet101
from utils import parse_args

class Eval():
    def __init__(self, checkpoint):
        self.checkpoint = torch.load(checkpoint, map_location=device)
        print('loading model: {}...'.format(checkpoint))
        # self.model = self.checkpoint['model'].module
        self.args = parse_args()
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

    def gen_feature(self, folder, file_name):
        print('gen features {}...'.format(folder))
        files = os.listdir(folder)

        features = {}
        with torch.no_grad():
            for file in files:
                print(file)
                name = file.split('.')[0].replace('_cmt', '')
                raw = cv2.imread(os.path.join(folder, file), True)
                img = self.get_image(raw, self.transformer).unsqueeze(0)
                feature = self.model(img.to(device)).cpu().numpy().squeeze(0)
                feature = feature/np.linalg.norm(feature)
                features[name] = feature.tolist()

        with open('./features/' + file_name, 'w') as fp:
            json.dump(features, fp)

        print(len(features.keys()))

    def get_image(self, img, transformer):
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = transformer(img)
        return img.to(device)

    def save_new_model(self):
        torch.save(self.model.state_dict(), './checkpoint/BEST_checkpoint_r101_new.pth')

    def load_file(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        return data

    def check_accuracy(self, id_card, people):
        match = 0
        non_match = 0
        non_match_list = []
        theta_list = []
        for id in id_card.keys():
            print(id)
            check = (id, '')
            min = 180
            id_feature = np.asarray(id_card[id])
            for person in people.keys():
                person_feature = np.asarray(people[person])
                cosine = np.dot(id_feature, person_feature)
                cosine = np.clip(cosine, -1.0, 1.0)
                theta = math.acos(cosine)
                theta = abs(theta * 180 / math.pi)
                if theta < min:
                    check = (id, person)
                    min = theta

            theta_list.append(min)

            if check[0] == check[1]:
                match += 1
            else:
                non_match += 1
                non_match_list.append(check)

            print(check)

        return match, non_match, non_match_list, theta_list

    def check_verification(self, id_card, people, threshold = 65.5):
        match = 0
        results = {}
        for id in id_card.keys():
            print(id)
            results[id] = []
            id_feature = np.asarray(id_card[id])
            for person in people.keys():
                person_feature = np.asarray(people[person])
                cosine = np.dot(id_feature, person_feature)
                cosine = np.clip(cosine, -1.0, 1.0)
                theta = math.acos(cosine)
                theta = abs(theta * 180 / math.pi)
                if theta < threshold:
                    match += 1
                    results[id].append(person)

        match = 0
        wrong_match = 0
        non_match = 0
        for key, value in results.items():
            if len(value) == 0:
                non_match += 1
            elif len(value) == 1:
                match += 1
            else:
                wrong_match += 1

        print(match)
        print(wrong_match)
        print(non_match)
        # print(results)

        return results

    def check_verification_euclid(self, id_card, people, threshold = 1.09):
        match = 0
        results = {}
        dist_list = []
        for id in id_card.keys():
            print(id)
            results[id] = []
            id_feature = np.asarray(id_card[id])
            for person in people.keys():
                person_feature = np.asarray(people[person])
                dist = np.linalg.norm(id_feature - person_feature)
                if dist <= threshold:
                    dist_list.append(dist)
                    results[id].append(person)

        match = 0
        wrong_match = 0
        non_match = 0
        for key, value in results.items():
            if len(value) == 0:
                non_match += 1
            elif len(value) == 1:
                match += 1
            else:
                wrong_match += 1

        print(match)
        print(wrong_match)
        print(non_match)
        print(sum(dist_list)/len(dist_list))

        return results


if __name__ == '__main__':
    checkpoint = './checkpoint/BEST_checkpoint_r101_new.pth'
    eval_face = Eval(checkpoint)
    # eval_face.gen_feature('./ekyc-face-126-cbnv/people_crop', 'people.json')


    id_card = eval_face.load_file('./features/id_card_old.json')
    people = eval_face.load_file('./features/people_old.json')
    match, non_match, non_match_list, theta_list = eval_face.check_accuracy(id_card, people)
    results = eval_face.check_verification_euclid(id_card, people)


