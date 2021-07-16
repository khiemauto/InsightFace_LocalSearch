import json
import yaml
from core import share_param
import numpy as np
import cv2
import base64
from datetime import datetime
    
soap_format = '''<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:esm="http://esmac.ewallet.lpb.com" xmlns:xsd="http://request.showroom.ewallet.lpb.com/xsd" xmlns:xsd1="http://common.entity.ewallet.lpb.com/xsd">
   <soapenv:Header/>
   <soapenv:Body>
      <esm:getSmartCustVip>
         <!--Optional:-->
         <esm:request>
            <!--Optional:-->
            <xsd:header>
               <!--Optional:-->
               <xsd1:channelCode>M</xsd1:channelCode>
               <!--Optional:-->
               <xsd1:deviceId>hungdv</xsd1:deviceId>
               <!--Optional:-->
               <xsd1:ip>127.0.0.1</xsd1:ip>
               <!--Optional:-->
               <xsd1:txnId>{txnId}</xsd1:txnId>
               <!--Optional:-->
               <xsd1:txnTime>{txnTime}</xsd1:txnTime>
               <!--Optional:-->
               <xsd1:userName>hungdv</xsd1:userName>
            </xsd:header>
            <!--Zero or more repetitions:-->
            <xsd:imgBase64>{base64}</xsd:imgBase64>
         </esm:request>
      </esm:getSmartCustVip>
   </soapenv:Body>
</soapenv:Envelope>'''

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

def add_redis_queue(img: np.ndarray):
    while share_param.redis_queue.qsize() > share_param.REDIS_QUEUE_SIZE*share_param.batch_size:
            share_param.redis_queue.get()
    share_param.redis_queue.put(img)

def opencv_to_base64(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        raise ValueError("image empty!")
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    retval, buffer = cv2.imencode(".jpg", image)
    bas64img = base64.b64encode(buffer)
    return bas64img

def get_soap_message(base64_img: str) -> str:
    txnId = datetime.now().strftime("%Y%m%d%H%M%S")
    txnTime = txnId
    soap_message = soap_format.format(txnId=txnId, txnTime=txnTime, base64=base64_img)
    return soap_message