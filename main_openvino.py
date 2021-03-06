import time
from datetime import datetime
import threading
import queue
import os
from datetime import datetime
import cv2
import numpy as np
import redis

from core import support, share_param
from insight_face_openvino.modules.tracking.custom_tracking import TrackingMultiCam
from insight_face_openvino.utils.database import FaceRecognitionSystem
from insight_face_openvino.modules.evalution.custom_evaluter import CustomEvaluter
import csv
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.FileHandler("main.log")        
handler.setFormatter(formatter)

main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.DEBUG)
main_logger.addHandler(handler)

def cam_thread_fun(deviceID: int, camURL: str):
    cap = cv2.VideoCapture(camURL)

    if not cap or not cap.isOpened():
        main_logger.warning(f"Camera not open {camURL}")
    else:
        main_logger.info(f"Camera opened {camURL}")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        main_logger.info(f"FPS of camera {deviceID}: {cap.get(cv2.CAP_PROP_FPS)}")

    FrameID = 1
    timeStep = 1/50  # 20FPS
    lastFrame = time.time()
    lastGood = time.time()

    while not share_param.bExit:
        time.sleep(0.001)
        if time.time()-lastGood>300:
            main_logger.info(f"Restart camera {camURL}")
            cap.open(camURL)
            lastGood = time.time()

        if cap is None or not cap.isOpened():
            continue

        if not share_param.bRunning or time.time()-lastFrame<timeStep:
            grabbed = cap.grab()
            if grabbed: lastGood = time.time()
            continue

        if time.time() - lastFrame > timeStep:
            lastFrame = time.time()
            grabbed, frame = cap.read()
            if grabbed: lastGood = time.time()
            if not grabbed or frame is None or frame.size==0:
                continue

        xstart = (frame.shape[1] - frame.shape[0])//2
        frame = frame[:, xstart: xstart + frame.shape[0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        support.add_cam_queue(deviceID, frame, FrameID)
        FrameID += 1

    if cap:
        cap.release()

def detect_thread_fun():
    totalTime = time.time()

    while not share_param.bExit: 
        if not share_param.bRunning:
            time.sleep(1)
            continue
        totalTime = time.time()
        time.sleep(0.001)

        if share_param.cam_queue.empty():
            continue

        raw_detect_inputs = []  # [[deviceId, rbg, frameID]]
        preTime = time.time()

        while len(raw_detect_inputs)<share_param.sdk_config["detector"]["batch_size"]:
            raw_detect_inputs.append(share_param.cam_queue.get())

        detect_inputs = []

        preTime = time.time()
        for id, (deviceId, rgb, FrameID) in enumerate(raw_detect_inputs):          
            detect_inputs.append(raw_detect_inputs[id])
        
        raw_detect_inputs.clear()
        del raw_detect_inputs

        if len(detect_inputs) == 0:
            continue

        preTime = time.time()
        rgbs = []
        
        for (deviceId, rgb, frameID) in detect_inputs:
            rgbs.append(rgb)

        preTime = time.time()
        bboxes_batch, landmarks_batch = share_param.facerec_system.sdk.detect_faces_batch(rgbs)
        print("Detect model Time:", time.time() - preTime)

        del rgbs

        for bboxes, landmarks, (deviceId, rgb, frameID) in zip(bboxes_batch, landmarks_batch, detect_inputs):
            main_logger.debug(f"Camera {deviceId} detected {len(bboxes)} faces with bboxes {bboxes}, landmarks {landmarks}")
            #Keep for recogn
            bbox_keeps = []
            landmark_keeps = []
            faceCropExpand_keeps = []

            #Draw tracking
            draw_bboxs = []
            draw_landmarks = []

            for bbox, landmark in zip(bboxes, landmarks):
                # cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                if bbox[0]<0 or bbox[1]<0 or bbox[2] > rgb.shape[1] or bbox[3] > rgb.shape[0]:
                    continue

                faceW = bbox[2] - bbox[0]
                faceH = bbox[3] - bbox[1]

                print(f"faceW {faceW}, faceH {faceH}")

                if faceW < share_param.sdk_config["detector"]["minface"] or faceH < share_param.sdk_config["detector"]["minface"]:
                    continue

                expandLeft = max(0, bbox[0] - faceW/3)
                expandTop = max(0, bbox[1] - faceH/3)
                expandRight = min(bbox[2] + faceW/3, rgb.shape[1])
                expandBottom = min(bbox[3] + faceH/3, rgb.shape[0])
                faceCropExpand = rgb[int(expandTop):int(expandBottom), int(expandLeft):int(expandRight)].copy()
                # faceCropExpand = rgb[int(expandTop):int(expandBottom), int(expandLeft):int(expandRight)]
                faceCropExpand_keeps.append(faceCropExpand)

                # draw_bboxs.append(bbox)
                # draw_landmarks.append(landmark)

                bbox_keeps.append(np.asarray(bbox))
                landmark_keeps.append(np.asarray(landmark))

            # support.add_imshow_queue(deviceId, rgb)
            
            data = (deviceId, bbox_keeps, landmark_keeps, faceCropExpand_keeps, rgb)
            support.add_detect_queue(data)

        main_logger.debug(f"Detect Time: {time.time() - totalTime}")

def recogn_thread_fun():
    totalTime = time.time()
    trackidtoname = {}      #{(deviceID,trackID): name}

    user_qualityscore_face_firsttime = {}  #{username:[facesize, blur, straight, firsttime, faceCropExpand, Pushed]}

    FPS = {}

    csvfile = open('blur.csv', 'w')
    spamwriter = csv.writer(csvfile, delimiter=',')
    # csvfile.close()

    while not share_param.bExit:
        if not share_param.bRunning:
            time.sleep(1)
            continue
        totalTime = time.time()
        time.sleep(0.001)

        for user in user_qualityscore_face_firsttime:
            if user_qualityscore_face_firsttime[user][5]:
                continue
            if time.time() - user_qualityscore_face_firsttime[user][3] > 1.0:
                filename = user + "G.jpg"
                photo_path = os.path.join("dataset/bestphotos", filename)
                cv2.imwrite(photo_path,  cv2.cvtColor(user_qualityscore_face_firsttime[user][4], cv2.COLOR_RGB2BGR))
                share_param.redisClient.lpush("image",support.opencv_to_base64(user_qualityscore_face_firsttime[user][4]))
                user_qualityscore_face_firsttime[user][4] = None
                user_qualityscore_face_firsttime[user][5] = True
                main_logger.info(f"Push face {user} to redis")

        if share_param.detect_queue.empty():
            continue

        recogn_inputs = []
        while not share_param.detect_queue.empty() and len(recogn_inputs)<share_param.sdk_config["embedder"]["batch_size"]:
            recogn_inputs.append(share_param.detect_queue.get())
        
        faceInfos = []      #FaceInformation of all frame in batch
        faceAligns = []     #Face Align of all frame in batch
        faceFrameInfos = {}     #Dict contain face infor and rgb of each frame in batch

        preTime = time.time()

        for iBuffer, (deviceId, bboxs, landmarks, faceCropExpands, rgb) in enumerate(recogn_inputs):
            if deviceId not in FPS:
                FPS[deviceId] = [20.0, time.time(), 0]
            else:
                if time.time()-FPS[deviceId][1] < 1.0:
                    FPS[deviceId][2] += 1
                else:
                    FPS[deviceId][0] = 0.9*FPS[deviceId][0] + 0.1*FPS[deviceId][2]
                    FPS[deviceId][2] = 0
                    FPS[deviceId][1] = time.time()
                    main_logger.debug(f"Pipleline FPS {deviceId}: {FPS[deviceId][0]}")

            faceFrameInfos[(iBuffer, deviceId)] = ([], rgb)     #Init faceFrameInfos
            
            for bbox, landmark, faceCropExpand in zip(bboxs, landmarks, faceCropExpands):
                if faceCropExpand is None or faceCropExpand.size ==0 or landmark is None or landmark.size==0:
                    continue
                faceAlign = share_param.facerec_system.sdk.align_face(rgb, landmark)
                faceInfos.append([deviceId, bbox, landmark, faceAlign, faceCropExpand, iBuffer])
                faceAligns.append(faceAlign)

        # print("Align Time:", time.time() - preTime)
        if len(faceAligns) > 0:
            preTime = time.time()
            embedder_batch_size = share_param.sdk_config["embedder"]["batch_size"]
            iLoop = len(faceAligns)//embedder_batch_size

            descriptors = []
            for i in range(iLoop):
                descriptor = share_param.facerec_system.sdk.get_descriptor_batch(faceAligns[i*embedder_batch_size: (i+1)*embedder_batch_size])
                descriptors.extend(descriptor)
            
            if len(faceAligns[iLoop*embedder_batch_size:]) > 0:
                descriptor = share_param.facerec_system.sdk.get_descriptor_batch(faceAligns[iLoop*embedder_batch_size:])
                descriptors.extend(descriptor)

            descriptors = np.array(descriptors)

            # print("faceAligns", len(faceAligns), "descriptors", len(descriptors))

            del faceAligns

            # print("Description Time:", time.time() - preTime)
            preTime = time.time()
            indicies = []
            distances = []
            if not share_param.facerec_system.photoid_to_username_photopath:
                for faceInfo, descriptor in zip(faceInfos, descriptors):
                    faceInfo.append(descriptor)
                    faceInfo.append('unknown')
                    faceInfo.append(0.0)
            else:
                indicies, distances = share_param.facerec_system.sdk.find_most_similar_batch(descriptors)
                for faceInfo, descriptor, indicie, distance in zip(faceInfos, descriptors, indicies, distances):
                    user_name = share_param.facerec_system.get_user_name(indicie[0])
                    faceInfo.append(descriptor)
                    faceInfo.append(user_name)
                    faceInfo.append(distance[0])
        
        for deviceId, bbox, landmark, faceAlign, faceCropExpand, iBuffer, descriptor, user_name, score in faceInfos:
            faceFrameInfos[(iBuffer, deviceId)][0].append([bbox, landmark, faceAlign, faceCropExpand, descriptor, user_name, score])

        # preTime = time.time()
        share_param.tracking_multiCam.update(faceFrameInfos)
        # print("Tracking time:", time.time() - preTime)

        for iBuffer, deviceId in faceFrameInfos:
            faceInfos = faceFrameInfos[(iBuffer, deviceId)][0]
            rgb = faceFrameInfos[(iBuffer, deviceId)][1]
            rgbDraw = rgb.copy()

            for bbox, landmark, faceAlign, faceCropExpand, descriptor, user_name, score, trackid, overlap in faceInfos:

                faceCrop = rgb[int(bbox[1]):int(bbox[3]),
                                    int(bbox[0]):int(bbox[2])]

                faceSize = float((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
                isillumination, threshillumination = share_param.evaluter_cams[deviceID].check_illumination(faceCrop)
                isNotBlur, threshnotblur = share_param.evaluter_cams[deviceID].check_not_blur(faceCrop)
                isStraightFace = share_param.evaluter_cams[deviceID].check_straight_face(rgb, landmark)
                
                if (deviceId,trackid) in trackidtoname:
                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {:03.3f} {:03.3f} {:03.3f} {}".format(trackid, trackidtoname[(deviceId,trackid)], score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if isStraightFace and isillumination and not overlap and threshnotblur > user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][1]:
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][0] = faceSize
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][1] = threshnotblur
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][4] = faceCropExpand

                        if user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][5] == True:
                            user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][5] = False
                            user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][3] = time.time()
                            main_logger.info(f"Found a better face of {trackidtoname[(deviceId,trackid)]}")

                
                elif score > share_param.dev_config["DEV"]["face_reg_score"]:
                    trackidtoname[(deviceId,trackid)] = user_name
                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {:03.3f} {:03.3f} {:03.3f} {}".format(trackid, user_name, score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if isStraightFace and isillumination and not overlap and threshnotblur > user_qualityscore_face_firsttime[user_name][1]:
                        user_qualityscore_face_firsttime[user_name][0] = faceSize
                        user_qualityscore_face_firsttime[user_name][1] = threshnotblur
                        user_qualityscore_face_firsttime[user_name][4] = faceCropExpand

                        if user_qualityscore_face_firsttime[user_name][5] == True:
                            user_qualityscore_face_firsttime[user_name][5] = False
                            user_qualityscore_face_firsttime[user_name][3] = time.time()
                            main_logger.info(f"Found a better face of {user_name}")

                else:
                    new_user_name = datetime.now().strftime("%H%M%S%f")
                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {:03.3f} {:03.3f} {:03.3f} {}".format(trackid, new_user_name, score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    if isNotBlur and isStraightFace and isillumination and not overlap:
                        trackidtoname[(deviceId,trackid)] = new_user_name
                        share_param.facerec_system.add_photo_descriptor_by_user_name(faceCropExpand, descriptor, new_user_name)
                        user_qualityscore_face_firsttime[new_user_name] = [faceSize, threshnotblur, isStraightFace, time.time(), faceCropExpand, False]
                        filename = new_user_name + "F.jpg"
                        photo_path = os.path.join("dataset/firstphotos", filename)
                        cv2.imwrite(photo_path, cv2.cvtColor(faceCropExpand, cv2.COLOR_RGB2BGR))
                        main_logger.info(f"Add new face {new_user_name}")
            
            support.add_imshow_queue(deviceId, rgbDraw)
        
        main_logger.debug(f"Recogn Time: {time.time() - totalTime}")
    csvfile.close()


def imshow_thread_fun():
    while not share_param.bExit:
        if not share_param.bRunning:
            time.sleep(1)
        else:
            time.sleep(0.01)
        while not share_param.imshow_queue.empty():
            title, image = share_param.imshow_queue.get()
            if share_param.dev_config["DEV"]["imshow"]:
                image = cv2.resize(image, (600,600))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image, time.strftime("%H:%M:%S"), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
                cv2.imshow(str(title), image)
            key = cv2.waitKey(10)

            if key == ord("q"):
                share_param.bExit = True
                main_logger.info(f"The shutdown command has been sent")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_logger.info("Starting application")
    main_logger.info("Reading configs")
    share_param.sdk_config = support.get_config_yaml("configs/sdk_config.yaml")
    share_param.dev_config = support.get_config_yaml("configs/dev_config.yaml")
    share_param.cam_infos = support.get_config_yaml("configs/cam_infos.yaml")    
    main_logger.info("Done reading configs")
    share_param.batch_size = len(share_param.cam_infos["CamInfos"])
    share_param.sdk_config["detector"]["batch_size"] = share_param.batch_size
    share_param.sdk_config["embedder"]["batch_size"] = share_param.batch_size*3
    main_logger.info("Init FaceRecognitionSystem")
    share_param.facerec_system = FaceRecognitionSystem(share_param.dev_config["DATA"]["photo_path"], share_param.sdk_config )
    main_logger.info("Init TrackingMultiCam")
    share_param.tracking_multiCam = TrackingMultiCam(share_param.sdk_config["tracking"])
    main_logger.info("Init RedisClient")
    share_param.redisClient = redis.StrictRedis(share_param.dev_config["REDIS"]["host"], share_param.dev_config["REDIS"]["port"])

    if share_param.dev_config["DATA"]["reload_database"]:
        share_param.facerec_system.create_database_from_folders(share_param.dev_config["DATA"]["photo_path"])
        share_param.facerec_system.save_database(share_param.dev_config["DATA"]["database_path"])
    share_param.facerec_system.load_database(share_param.dev_config["DATA"]["database_path"])

    share_param.cam_queue = queue.Queue(maxsize=share_param.CAM_QUEUE_SIZE*share_param.batch_size+1)
    share_param.detect_queue = queue.Queue(maxsize=share_param.DETECT_QUEUE_SIZE*share_param.batch_size+1)
    share_param.recogn_queue = queue.Queue(maxsize=share_param.RECOGN_QUEUE_SIZE*share_param.batch_size+1)
    share_param.imshow_queue = queue.Queue(maxsize=share_param.IMSHOW_QUEUE_SIZE*share_param.batch_size+1)

    for cam_info in share_param.cam_infos["CamInfos"]:
        deviceID = cam_info["DeviceID"]
        camURL = cam_info["LinkRTSP"]

        evalution_config = share_param.sdk_config["evaluter"]
        evalution_config["illumination_threshold"] = cam_info["illumination_threshold"]
        evalution_config["blur_threshold"] = cam_info["blur_threshold"]

        share_param.evaluter_cams[deviceID] = CustomEvaluter(evalution_config)
        share_param.cam_threads[deviceID] = threading.Thread(target=cam_thread_fun, daemon=True, args=(deviceID, camURL))

    share_param.detect_thread = threading.Thread(target=detect_thread_fun, daemon=True, args=())
    share_param.recogn_thread = threading.Thread(target=recogn_thread_fun, daemon=True, args=())
    share_param.imshow_thread = threading.Thread(target=imshow_thread_fun, daemon=True, args=())

    for deviceID in share_param.cam_threads:
        share_param.cam_threads[deviceID].start()
    main_logger.info(f"cam_threads started")
    share_param.detect_thread.start()
    main_logger.info(f"detect_thread started")
    share_param.recogn_thread.start()
    main_logger.info(f"recogn_thread started")
    share_param.imshow_thread.start()
    main_logger.info(f"imshow_thread started")
    share_param.imshow_thread.join()
    main_logger.info(f"imshow_thread quited")
    share_param.recogn_thread.join()
    main_logger.info(f"recogn_thread quited")
    share_param.detect_thread.join()
    main_logger.info(f"detect_thread quited")
    main_logger.info(f"Appication shutdown")