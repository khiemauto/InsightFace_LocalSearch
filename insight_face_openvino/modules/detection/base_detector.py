import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFaceDetector(ABC):
    """
    Base class for detection model
    """

    def __init__(self, config: dict):
        """
        Args:
            config: model config from module outside. Used to redefine inside config fields.
        """
        self.config = config

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess raw image of RGB format with shape (H, W, C) for model prediction.

        Args:
            image: image in RGB format. Shape: (H, W, C)

        Returns:
            np.ndarray: preprocessed image for prediction
        """
        raise NotImplementedError

    @abstractmethod
    def _predict_raw(self, image: np.ndarray) -> np.ndarray:
        """
        Make prediction on preprocessed image.

        Args:
            image: preprocessed image by _preprocess method.

        Returns:
            np.ndarray: raw prediction of model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to make prediction on raw image. Return processed detections

        Args:
            image: image in RGB format. Shape: (H, W, C)

        Returns:
            Tuple of bboxes and landmarks
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to postprocess model's raw prediction.

        Args:
            raw_prediction: model's raw prediction output

        Returns:
            Model's postprocessed output. Tuple of bboxes and landmarks
        """
        raise NotImplementedError
