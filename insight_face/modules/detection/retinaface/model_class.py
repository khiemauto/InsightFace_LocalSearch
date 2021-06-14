import torch
import numpy as np
import yaml
import cv2
from typing import List

from torch import Tensor
from pathlib import Path
from typing import Tuple

from ..base_detector import BaseFaceDetector
from .dependencies.retinaface import RetinaFace as ModelClass
from ....utils.load_utils import load_model
from .dependencies.utils import decode, decode_landm, py_cpu_nms
from .dependencies.prior_box import PriorBox

model_urls = {
    "res50": "https://bitbucket.org/khiembka1992/data/raw/abe66827e127477581587e7e90cdefaab459c426/InsightFace/Resnet50_Final.pth",
    "mnet1": "https://bitbucket.org/khiembka1992/data/raw/abe66827e127477581587e7e90cdefaab459c426/InsightFace/mobilenet0.25_Final.pth",
}

class RetinaFace(BaseFaceDetector):
    def __init__(self, config: dict):
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
        self.device = torch.device(self.config["device"])
        self.model = ModelClass(self.model_config, phase="test")
        self.model = load_model(self.model, model_urls[backbone], True if self.config["device"] == "cpu" else False)
        self.model.eval()
        self.model.to(self.device)
        self.model_input_shape = None
        self.resize_scale = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = np.float32(image)
        target_size = self.config["image_size"]
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[:2])
        im_size_max = np.max(im_shape[:2])
        resize = float(target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.config.get("origin_size", False):
            resize = 1

        self.resize_scale = resize
        if resize != 1:
            img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img -= np.asarray((104, 117, 123), dtype=np.float32)
        return img

    def _predict_raw(self, image: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        img = image.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        pred = self.model(img)
        # loc = loc.detach().cpu().numpy()
        # conf = conf.detach().cpu().numpy()
        # landms = landms.detach().cpu().numpy()
        return pred

    def _postprocess(self, raw_prediction: Tuple[Tensor, Tensor, Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        loc, conf, landms = raw_prediction
        img_h, img_w = self.model_input_shape[:2]
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        scale = scale.to(self.device)

        priorbox = PriorBox(self.model_config, image_size=(img_h, img_w))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.model_config["variance"])
        boxes = boxes * scale / self.resize_scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.model_config["variance"])
        scale1 = torch.tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h], dtype=torch.float)
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize_scale
        landms = landms.cpu().numpy()

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
            target_size = self.config["image_size"]
            max_size = 2150
            im_shape = img.shape
            im_size_min = np.min(im_shape[:2])
            im_size_max = np.max(im_shape[:2])
            resize = float(target_size) / float(im_size_min)

            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if self.config.get("origin_size", False):
                resize = 1

            self.resize_scale = resize
            if resize != 1:
                img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            img = img.astype(np.float32)
            img -= np.asarray((104, 117, 123), dtype=np.float32)
            imgs.append(img)
        return imgs

    def _predict_raw_batch(self, images: List[np.ndarray]) -> Tuple[Tensor, Tensor, Tensor]:
        imgs = []
        for image in images:
            img = image.transpose((2, 0, 1))
            imgs.append(torch.from_numpy(img))

        imgs = torch.stack(imgs).to(self.device)
        pred = self.model(imgs)
        return pred

    def _postprocess_batch(self, raw_predictions: List[Tuple[Tensor, Tensor, Tensor]], preprocess_sizes) -> List[Tuple[np.ndarray, np.ndarray]]:
        dets_batch = []
        landms_batch = []
        locs, confs, landmss = raw_predictions
        # print(len(locs), len(confs), len(landmss))
        for loc, conf, landms, preprocess_size in zip(locs, confs, landmss, preprocess_sizes):
            # print(len(raw_prediction[0]), len(raw_prediction[1]), len(raw_prediction[2]))
            # loc, conf, landms = raw_prediction
            # img_h, img_w = self.model_input_shape[:2]
            # scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
            # scale = scale.to(self.device)

            priorbox = PriorBox(self.model_config, image_size=(preprocess_size[0], preprocess_size[1]))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.model_config["variance"])
            # boxes = boxes * scale / self.resize_scale
            boxes = boxes.cpu().numpy()

            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landms = decode_landm(landms.data.squeeze(0), prior_data, self.model_config["variance"])
            # scale1 = torch.tensor([preprocess_size[1], preprocess_size[0], preprocess_size[1], preprocess_size[0], preprocess_size[1], preprocess_size[0], preprocess_size[1], preprocess_size[0], preprocess_size[1], preprocess_size[0]], dtype=torch.float)
            # scale1 = scale1.to(self.device)
            # landms = landms * scale1 / self.resize_scale
            landms = landms.cpu().numpy()

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

    # def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    #     images = self._preprocess_batch(images)
    #     self.model_input_shape = images[0].shape
    #     raw_preds = self._predict_raw_batch(images)
    #     bboxes_batch, landms_batch = self._postprocess_batch(raw_preds)

    #     # print(len(bboxes_batch), len(landms_batch))

    #     landmarks_batch = []
    #     for bboxes, landms in zip(bboxes_batch, landms_batch):
    #         converted_landmarks = []
    #         # convert to our landmark format (2,5)
    #         for landmarks_set in landms:
    #             x_landmarks = []
    #             y_landmarks = []
    #             for i, lm in enumerate(landmarks_set):
    #                 if i % 2 == 0:
    #                     x_landmarks.append(lm)
    #                 else:
    #                     y_landmarks.append(lm)
    #             converted_landmarks.append(x_landmarks + y_landmarks)

    #         landmarks_batch.append(np.array(converted_landmarks))

    #     return bboxes_batch, landmarks_batch



    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        origin_sizes = []
        for image in images:
            origin_sizes.append(image.shape[:2])

        images = self._preprocess_batch(images)
        preprocess_sizes = []
        for image in images:
            preprocess_sizes.append(image.shape[:2])

        # self.model_input_shape = images[0].shape
        raw_preds = self._predict_raw_batch(images)
        bboxes_batch, landms_batch = self._postprocess_batch(raw_preds, preprocess_sizes)

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