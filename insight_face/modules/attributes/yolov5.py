import argparse
import sys
import time
from pathlib import Path
from typing import List
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import ndarray, random
import yaml
import numpy as np
import os

from .models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
from .utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from .utils.plots import plot_one_box
# from .utils.torch_utils import select_device, load_classifier
# from core import support
# from insight_face.modules.detection.retinaface.model_class import RetinaFace
from ...utils.load_utils import get_file_from_url

model_urls = {
    "yolov5s": "https://bitbucket.org/khiembka1992/data/raw/891d568aab90ef9178666c442df6392640cdf8a4/InsightFace/yolov5_face_attrs.pt"
}

class FaceAttributes():
    def __init__(self, config) -> None:
        self.config = config
        self.device = config["device"]
        architecture = config["architecture"]

        path_to_model_config = Path(Path(__file__).parent, "config.yaml").as_posix()
        with open(path_to_model_config, "r") as fp:
            self.model_config = yaml.load(fp, Loader=yaml.FullLoader)

        if architecture not in self.model_config.keys():
            raise ValueError(f"Unsupported backbone: {architecture}!")

        self.model_config = self.model_config[architecture]

        pt_file = get_file_from_url(url=model_urls[architecture], model_dir=os.path.dirname(self.model_config["weights"]), progress=True, unzip=False)

        # print(self.model_config)
        # self.model = torch.load(self.model_config["weights"], map_location=self.device)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', pt_file).to(self.device).eval()
        # print(self.model)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(self.names)


    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def bb_intersection_per_first(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea)

        # return the intersection over union value
        return iou

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        # face_tensor = self.preprocess(face)
        self.model_config["img_size"]
        stride = int(self.model.stride.max())
        resize = (self.model_config["img_size"] // stride) * stride
        face = cv2.resize(face, (resize, resize))
        face = face.transpose((2, 0, 1))
        return face

    def detect(self, image: np.ndarray):
        face_tensor = self._preprocess(image)
        # print(face_tensor.shape)

        # Get names and colors

        t0 = time.time()
        img = torch.from_numpy(face_tensor).to(self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        preTime = time.time()
        pred = self.model(img, augment=self.model_config["augment"])[0]
        # print(f'Attributes: {time.time() - preTime:.3f}')

        # Apply NMS
        pred = non_max_suppression(pred, self.model_config["conf_thres"], self.model_config["iou_thres"], agnostic=self.model_config["agnostic_nms"])[0]
        # t2 = time_synchronized()
        # print(pred)
        names = []
        for name in pred:
            names.append(self.names[int(name[5])])
        return names

    def detect_batch(self, images: List[np.ndarray]):
        imgs = []
        for image in images:
            face_tensor = self._preprocess(image)
            img = torch.from_numpy(face_tensor).float()
            img /= 255.0
            imgs.append(img)

        imgs = torch.stack(imgs).to(self.device)

        # Inference
        preTime = time.time()
        pred = self.model(imgs, augment=self.model_config["augment"])[0]
        # print(f'Attributes: {time.time() - preTime:.3f}')

        # Apply NMS
        preds = non_max_suppression(pred, self.model_config["conf_thres"], self.model_config["iou_thres"], agnostic=self.model_config["agnostic_nms"])
        attrs_batch = []
        # print(preds)
        for pred in preds:
            attrs = set()
            for attr in pred:
                attrs.add(self.names[int(attr[5])])
            attrs_batch.append(sorted(attrs))
        return attrs_batch