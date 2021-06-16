from typing import List
import numpy as np
from pathlib import Path
import yaml
import os

from ..base_embedder import BaseFaceEmbedder
from ....utils.load_utils import get_file_from_url

model_urls = {
    "iresnet34": "https://bitbucket.org/khiembka1992/data/raw/1b859b3a22ade2e822aab481a2ecc7334026c078/OpenVino/iresnet34",
    "iresnet50": "https://bitbucket.org/khiembka1992/data/raw/1b859b3a22ade2e822aab481a2ecc7334026c078/OpenVino/iresnet50",
    "iresnet100": "https://bitbucket.org/khiembka1992/data/raw/1b859b3a22ade2e822aab481a2ecc7334026c078/OpenVino/iresnet100"
}
class InsightFaceEmbedder(BaseFaceEmbedder):
    """Implements inference of face recognition nets from InsightFace project."""
    def __init__(self, ie, config: dict):
        self.config = config
        self.device = config["device"]
        architecture = config["architecture"]

        path_to_model_config = Path(Path(__file__).parent, "config.yaml").as_posix()
        with open(path_to_model_config, "r") as fp:
            self.model_config = yaml.load(fp, Loader=yaml.FullLoader)

        if architecture not in self.model_config.keys():
            raise ValueError(f"Unsupported backbone: {architecture}!")

        self.model_config = self.model_config[architecture]
        
        xml_file = get_file_from_url(url=model_urls[architecture]+".xml", model_dir=os.path.dirname(self.model_config["weights_path"]))
        bin_file = get_file_from_url(url=model_urls[architecture]+".bin", model_dir=os.path.dirname(self.model_config["weights_path"]))
        self.net = ie.read_network(xml_file, bin_file)
        self.batch_size = config["batch_size"]
        self.net.batch_size = self.batch_size
        self.input_name = self.model_config["input_name"]
        self.output_name = self.model_config["output_name"]
        self.embedder = ie.load_network(network=self.net, device_name="CPU", config={"DYN_BATCH_ENABLED": "YES"})
        self.mean = [0.5] * 3
        self.std = [0.5 * 256 / 255] * 3

    def preprocess(self, face:np.ndarray) -> np.ndarray:
        face = face.astype(np.float32)/255.0
        face = (face-self.mean)/self.std
        face = face.transpose((2, 0, 1))
        return face

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        face_tensor = self.preprocess(face)
        face_tensor = np.expand_dims(face_tensor, axis=0)
        return face_tensor

    def _predict_raw(self, face: np.ndarray) -> np.ndarray:
        self.embedder.requests[0].infer(inputs= {self.input_name: face})
        features = self.embedder.requests[0].outputs[self.output_name]
        return features

    def _postprocess(self, raw_prediction: np.ndarray) -> np.ndarray:
        descriptor = raw_prediction[0]
        descriptor = descriptor / np.linalg.norm(descriptor)
        return descriptor

    def _get_raw_model(self):
        return self.embedder

    #Batch run
    def _preprocess_batch(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        face_tensors = []
        for face in faces:
            face_tensors.append(self.preprocess(face))
        return face_tensors

    def _predict_raw_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        if len(faces) > self.batch_size:
            raise NotImplementedError
        input_raw = np.zeros(shape=[self.batch_size, 3, self.config["image_size"], self.config["image_size"]], dtype=np.float32)
        for i, face in enumerate(faces):
            input_raw[i] = face
        self.embedder.requests[0].set_batch(len(faces))
        self.embedder.requests[0].infer(inputs= {self.input_name: input_raw})
        features = self.embedder.requests[0].outputs[self.output_name][:len(faces)]
        return features

    def _postprocess_batch(self, raw_predictions: np.ndarray) -> np.ndarray:
        descriptors = raw_predictions / np.linalg.norm(raw_predictions, axis=1)[:,None]
        return descriptors
    
    def run_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if len(images) == 0:
            return []
        return self._postprocess_batch(self._predict_raw_batch(self._preprocess_batch(images)))