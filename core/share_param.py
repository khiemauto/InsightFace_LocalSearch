dev_config = None
sdk_config = None
cam_infos = None
facerec_system = None

tracking_multiCam = None
evaluter_cams = {}
redisClient = None

batch_size = 5

CAM_QUEUE_SIZE = 5
DETECT_QUEUE_SIZE = 5
RECOGN_QUEUE_SIZE = 10
IMSHOW_QUEUE_SIZE = 5
REDIS_QUEUE_SIZE = 10
SAYNAME_QUEUE_SIZE = 20

bRunning = True
bExit = False
cam_infos = {}
cam_threads = {}
detect_thread = None
recogn_thread = None
imshow_thread = None
redis_thread = None
api_thread = None

cam_queue = None
detect_queue = None
recogn_queue = None
imshow_queue = None
redis_queue = None
sayname_queue = None

#Hyperbol blur
qi = 1345.33325
b = 0.52109685
di = 2.3316e-04