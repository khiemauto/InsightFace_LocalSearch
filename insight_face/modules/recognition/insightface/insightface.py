from typing import List
import numpy as np
import torch
from torchvision import transforms

from .nets import iresnet

from ..base_embedder import BaseFaceEmbedder


class InsightFaceEmbedder(BaseFaceEmbedder):

    """Implements inference of face recognition nets from InsightFace project."""

    def __init__(self, config: dict):

        self.device = config["device"]
        architecture = config["architecture"]

        if architecture == "iresnet34":
            self.embedder = iresnet.iresnet34(pretrained=True)
        elif architecture == "iresnet50":
            self.embedder = iresnet.iresnet50(pretrained=True)
        elif architecture == "iresnet100":
            self.embedder = iresnet.iresnet100(pretrained=True)
        else:
            raise ValueError(f"Unsupported network architecture: {architecture}")

        self.embedder.eval()
        self.embedder.to(self.device)

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3

        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        face_tensor = self.preprocess(face).unsqueeze(0).to(self.device)
        return face_tensor

    def _predict_raw(self, face: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            features = self.embedder(face)
        return features

    def _postprocess(self, raw_prediction: np.ndarray) -> np.ndarray:
        descriptor = raw_prediction[0].cpu().numpy()
        descriptor = descriptor / np.linalg.norm(descriptor)
        return descriptor

    def _get_raw_model(self):
        return self.embedder

    #Batch run
    def _preprocess_batch(self, faces: List[np.ndarray]) -> List[torch.Tensor]:
        face_tensors = []
        for face in faces:
            face_tensors.append(self.preprocess(face))
        return face_tensors

    def _predict_raw_batch(self, faces: List[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            faces = torch.stack(faces).to(self.device)
            features = self.embedder(faces)
        return features

    def _postprocess_batch(self, raw_predictions: torch.Tensor) -> np.ndarray:
        raw_predictions = raw_predictions.cpu().numpy()
        descriptors = raw_predictions / np.linalg.norm(raw_predictions, axis=1)[:,None]
        return descriptors
    
    def run_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return self._postprocess_batch(self._predict_raw_batch(self._preprocess_batch(images)))