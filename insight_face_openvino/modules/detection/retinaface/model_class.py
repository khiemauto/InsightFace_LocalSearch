import torch
import numpy as np
import yaml
import cv2
from typing import List

from torch import Tensor
from pathlib import Path
from typing import Tuple
import time

from ..base_detector import BaseFaceDetector
from .dependencies.utils import decode, decode_landm, py_cpu_nms
from .dependencies.prior_box import PriorBox

model_urls = {
    "res50": "https://bitbucket.org/khiembka1992/data/raw/abe66827e127477581587e7e90cdefaab459c426/InsightFace/Resnet50_Final.pth",
    "mnet1": "https://bitbucket.org/khiembka1992/data/raw/abe66827e127477581587e7e90cdefaab459c426/InsightFace/mobilenet0.25_Final.pth",
}

class RetinaFace(BaseFaceDetector):
    def __init__(self, ie, config: dict):
        """
        Args:
            config: detector config for configuration of model from outside
        """
        super().__init__(config)
        backbone = config["architecture"]
        path_to_model_config = Path(Path(__file__).parent, "config.yaml").as_posix()
        with open(path_to_model_config, "r") as fp:
            self.model_config = yaml.load(fp, Loader=yaml.FullLoader)

        if backbone not in self.model_config.keys():
            raise ValueError(f"Unsupported backbone: {backbone}!")

        self.model_config = self.model_config[backbone]

        self.net = ie.read_network(self.model_config["weights_path"] + ".xml", self.model_config["weights_path"] + ".bin")
        self.batch_size = config["batch_size"]
        self.net.batch_size = self.batch_size
        self.input_name = self.model_config["input_name"]
        self.output_names = self.model_config["output_names"]
        self.model = ie.load_network(network=self.net, device_name="CPU")

        priorbox = PriorBox(self.model_config, image_size=(self.config["image_size"], self.config["image_size"]))
        self.prior_data = priorbox.forward()

        self.model_input_shape = None
        self.resize_scale = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = np.float32(image)
        target_size = self.config["image_size"]
        im_shape = img.shape
        im_shape = img.shape
        img_width = im_shape[1]
        img_height = im_shape[0]

        self.resize_scale = [float(target_size) / float(img_width), float(target_size) / float(img_height)]
        img = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))

        img = img.astype(np.float32)
        img -= np.asarray((104, 117, 123), dtype=np.float32)
        return img

    def _predict_raw(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = image.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        self.model.requests[0].infer(inputs= {self.input_name: img})
        pred = self.model.requests[0].outputs

        pred = (pred[self.output_names[0]],pred[self.output_names[1]], pred[self.output_names[2]])
        return pred

    def _postprocess(self, raw_prediction: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        loc, conf, landms = raw_prediction
        img_h, img_w = self.model_input_shape[:2]
        scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        boxes = decode(loc, self.prior_data, self.model_config["variance"])
        resize_scale = np.array(self.resize_scale*2, dtype=np.float32)
        boxes = boxes * scale / resize_scale

        scores = conf[:, 1]

        landms = decode_landm(landms, self.prior_data, self.model_config["variance"])
        scale1 = np.array([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h], dtype=np.float32)
        resize_scale = np.array(self.resize_scale*5, dtype=np.float32)
        landms = landms * scale1 / resize_scale

        # ignore low scores
        inds = np.where(scores > self.config["conf_threshold"])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config["nms_threshold"])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        return dets, landms

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = self._preprocess(image)
        self.model_input_shape = image.shape
        raw_pred = self._predict_raw(image)
        bboxes, landms = self._postprocess(raw_pred)

        converted_landmarks = []
        # convert to our landmark format (2,5)
        for landmarks_set in landms:
            x_landmarks = []
            y_landmarks = []
            for i, lm in enumerate(landmarks_set):
                if i % 2 == 0:
                    x_landmarks.append(lm)
                else:
                    y_landmarks.append(lm)
            converted_landmarks.append(x_landmarks + y_landmarks)

        landmarks = np.array(converted_landmarks)

        return bboxes, landmarks

    def _get_raw_model(self):
        return self.model

    #Batch run
    def _preprocess_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        imgs = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = np.float32(image)


            img = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))

            img = img.astype(np.float32)
            img -= np.asarray((104, 117, 123), dtype=np.float32)
            imgs.append(img)
        return imgs


    def _predict_raw_batch(self, images: List[np.ndarray]) -> Tuple[Tensor, Tensor, Tensor]:
        if len(images) > self.batch_size:
            raise NotImplementedError
        imgs = np.zeros(shape=(self.batch_size, 3, self.config["image_size"], self.config["image_size"]), dtype=np.float32)
        for i, image in enumerate(images):
            img = image.transpose((2, 0, 1))
            imgs[i] = img

        self.model.requests[0].infer(inputs= {self.input_name: imgs})
        pred = self.model.requests[0].outputs

        pred = (pred[self.output_names[0]][:len(images)],pred[self.output_names[1]][:len(images)], pred[self.output_names[2]][:len(images)])
        return pred

    def _postprocess_batch(self, raw_predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        dets_batch = []
        landms_batch = []
        locs, confs, landmss = raw_predictions
        # print(len(locs), len(confs), len(landmss))
        for loc, conf, landms in zip(locs, confs, landmss):
            boxes = decode(loc, self.prior_data, self.model_config["variance"])

            scores = conf[:, 1]

            landms = decode_landm(landms, self.prior_data, self.model_config["variance"])

            # ignore low scores
            inds = np.where(scores > self.config["conf_threshold"])[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.config["nms_threshold"])
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            dets_batch.append(dets)
            landms_batch.append(landms)

        return dets_batch, landms_batch

    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        origin_sizes = []
        for image in images:
            origin_sizes.append(image.shape[:2])

        images = self._preprocess_batch(images)
        preprocess_sizes = []
        for image in images:
            preprocess_sizes.append(image.shape[:2])

        raw_preds = self._predict_raw_batch(images)
        bboxes_batch, landms_batch = self._postprocess_batch(raw_preds)

        unscale_bboxes_batch = []
        unscale_landms_batch = []
        for origin_size, boxes, landms in zip(origin_sizes, bboxes_batch, landms_batch):
            box_scale = np.array([origin_size[1], origin_size[0], origin_size[1], origin_size[0], 1], dtype=np.float32)
            landmark_scale = np.array([origin_size[1], origin_size[0], origin_size[1], origin_size[0], origin_size[1], origin_size[0], origin_size[1], origin_size[0], origin_size[1], origin_size[0]], dtype=np.float32)
            boxes = boxes * box_scale
            landms = landms * landmark_scale
            unscale_bboxes_batch.append(boxes)
            unscale_landms_batch.append(landms)

        landmarks_batch = []
        for landms in unscale_landms_batch:
            converted_landmarks = []
            # convert to our landmark format (2,5)
            for landmarks_set in landms:
                x_landmarks = []
                y_landmarks = []
                for i, lm in enumerate(landmarks_set):
                    if i % 2 == 0:
                        x_landmarks.append(lm)
                    else:
                        y_landmarks.append(lm)
                converted_landmarks.append(x_landmarks + y_landmarks)

            landmarks_batch.append(np.array(converted_landmarks))

        return unscale_bboxes_batch, landmarks_batch