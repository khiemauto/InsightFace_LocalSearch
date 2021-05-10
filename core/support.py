import json
import yaml
from core import share_param
import numpy as np
import cv2
import base64
    
def get_config_json(local_file) -> json:
    """
    Get deverlop config
    :local_file: dev config file
    :return: json config
    """
    res = None
    try:
        with open(local_file, 'r') as json_file:
            res = json.load(json_file)
    except:
        print(f"[Error] Read json {local_file}")
    return res

def get_config_yaml(local_file) -> json:
    """
    Get deverlop config
    :local_file: dev config file
    :return: json config
    """
    res = None
    try:
        with open(local_file, 'r') as fp:
            res = yaml.load(fp, Loader=yaml.FullLoader)
    except:
        print(f"[Error] Read yaml {local_file}")
    return res


def custom_imshow(title: str, image: np.ndarray):
    if share_param.devconfig["DEV"]["imshow"]:
        cv2.imshow(title, image)
        cv2.waitKey(1)

def add_imshow_queue(deviceID: int, img: np.ndarray):
    while share_param.imshow_queue.qsize() > share_param.IMSHOW_QUEUE_SIZE*share_param.batch_size:
            share_param.imshow_queue.get()
    share_param.imshow_queue.put((deviceID, img))

def add_cam_queue(deviceID: int, img: np.ndarray, FrameID: int):
    while share_param.cam_queue.qsize() > share_param.CAM_QUEUE_SIZE*share_param.batch_size:
            share_param.cam_queue.get()
    share_param.cam_queue.put((deviceID, img, FrameID))

def add_detect_queue(data):
    while share_param.detect_queue.qsize() > share_param.DETECT_QUEUE_SIZE*share_param.batch_size:
        share_param.detect_queue.get()
    share_param.detect_queue.put(data)

def opencv_to_base64(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        raise ValueError("image empty!")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    retval, buffer = cv2.imencode(".jpg", image)
    bas64img = base64.b64encode(buffer)
    return bas64img
