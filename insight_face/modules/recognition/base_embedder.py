import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFaceEmbedder(ABC):
    """
    Base class for extracting face descriptors from images.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self._postprocess(self._predict_raw(self._preprocess(image)))

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess raw image of RGB format with shape (H, W, C) for model prediction."""
        raise NotImplementedError

    @abstractmethod
    def _predict_raw(self, image: np.ndarray) -> np.ndarray:
        """Get model output on preprocessed image."""
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, raw_prediction: np.ndarray) -> np.ndarray:
        """Postprocess model output."""
        raise NotImplementedError
