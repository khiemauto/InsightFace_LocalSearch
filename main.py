import time
from datetime import datetime
import threading
import queue
import os
from datetime import datetime
import cv2
import numpy as np
import uvicorn
from core.rest import FaceRecogAPI

from core import support, share_param
from insight_face.modules.tracking.custom_tracking import TrackingMultiCam
from insight_face.utils.database import FaceRecognitionSystem
from insight_face.modules.evalution.custom_evaluter import CustomEvaluter
# import csv
import logging
import requests

import xml.etree.ElementTree as ET

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.FileHandler("main.log")        
handler.setFormatter(formatter)

main_logger = logging.getLogger(__name__)
# main_logger.setLevel(logging.DEBUG)
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

        while (not share_param.cam_queue.empty()) and len(raw_detect_inputs)<share_param.batch_size:
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
        # print("Detect Time:", time.time() - totalTime)

        del rgbs

        for bboxes, landmarks, (deviceId, rgb, frameID) in zip(bboxes_batch, landmarks_batch, detect_inputs):
            # main_logger.debug(f"Camera {deviceId} detected {len(bboxes)} faces with bboxes {bboxes}, landmarks {landmarks}")
            #Keep for recogn
            bbox_keeps = []
            landmark_keeps = []
            faceCropExpand_keeps = []

            #Draw tracking
            draw_bboxs = []
            draw_landmarks = []

            for bbox, landmark in zip(bboxes, landmarks):
                if bbox[0]<0 or bbox[1]<0 or bbox[2] > rgb.shape[1] or bbox[3] > rgb.shape[0]:
                    continue

                faceW = bbox[2] - bbox[0]
                faceH = bbox[3] - bbox[1]

                # print(f"faceW {faceW}, faceH {faceH}")

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
            
            # print("bbox_keeps", bbox_keeps)
            post_bboxes_batch, post_landmarks_batch = [], []

            for faceCropExpand_keep in faceCropExpand_keeps:
                post_bboxes, post_landmarks = share_param.facerec_system.sdk.detect_post_faces(faceCropExpand_keep)
                post_bboxes_batch.append(post_bboxes)
                post_landmarks_batch.append(post_landmarks)
            # print("post_bboxes_batch", post_bboxes_batch)

            post_bbox_keeps = []
            post_landmark_keeps = []
            post_faceCropExpand_keeps = []
            
            for i, (post_bboxes, post_landmarks) in enumerate(zip(post_bboxes_batch, post_landmarks_batch)):
                if len(post_bboxes) == 0:
                    cv2.imwrite(f"dataset/detect_post_ignore/{time.time()}.jpg", faceCropExpand_keeps[i])
                    print("detect_post ignore face")
                    continue

                post_bbox_keeps.append(bbox_keeps[i])
                post_landmark_keeps.append(landmark_keeps[i])
                post_faceCropExpand_keeps.append(faceCropExpand_keeps[i])

            data = (deviceId, post_bbox_keeps, post_landmark_keeps, post_faceCropExpand_keeps, rgb)
            # data = (deviceId, bbox_keeps, landmark_keeps, faceCropExpand_keeps, rgb)
            support.add_detect_queue(data)

        main_logger.debug(f"Detect Time: {time.time() - totalTime}")

user_qualityscore_face_firsttime = {}  #{username:[facesize, blur, straight, firsttime, faceCropExpand, Pushed, lastSeeTime, customerName]}

def recogn_thread_fun():

    global user_qualityscore_face_firsttime
    totalTime = time.time()
    trackidtoname = {}      #{(deviceID,trackID): name}

    # user_qualityscore_face_firsttime = {}  #{username:[facesize, blur, straight, firsttime, faceCropExpand, Pushed, lastSeeTime, customerName, numPush]}

    FPS = {}

    # csvfile = open('blur.csv', 'w')
    # spamwriter = csv.writer(csvfile, delimiter=',')
    # csvfile.close()

    while not share_param.bExit:
        if not share_param.bRunning:
            time.sleep(1)
            continue
        totalTime = time.time()
        time.sleep(0.001)

        for user in list(user_qualityscore_face_firsttime):
            if time.time() - user_qualityscore_face_firsttime[user][6] > 1800.0:
                share_param.facerec_system.del_photo_by_user_name(user)
                del user_qualityscore_face_firsttime[user]
                trackidtoname = { k:v for k, v in trackidtoname.items() if v!=user }
                continue

            #Check new request
            if user_qualityscore_face_firsttime[user][5]:
                continue
            #Check if user already has an customer name?
            if user_qualityscore_face_firsttime[user][7] != "":
                continue

            if user_qualityscore_face_firsttime[user][8] >3:
                continue

            #Check time from the last best face.
            if time.time() - user_qualityscore_face_firsttime[user][3] < 2.0:
                continue

            filename = user + "G.jpg"
            photo_path = os.path.join("dataset/bestphotos", filename)
            equ = cv2.cvtColor(user_qualityscore_face_firsttime[user][4], cv2.COLOR_RGB2BGR)

            cv2.imwrite(photo_path,  equ)
            support.add_redis_queue(user, equ)
            
            user_qualityscore_face_firsttime[user][4] = None
            user_qualityscore_face_firsttime[user][5] = True
            user_qualityscore_face_firsttime[user][8] += 1
            main_logger.info(f"Push face {user} to redis")

        if share_param.detect_queue.empty():
            continue

        recogn_inputs = []
        # while not share_param.detect_queue.empty():
        while not share_param.detect_queue.empty() and len(recogn_inputs)<share_param.batch_size:
            recogn_inputs.append(share_param.detect_queue.get())
        
        faceInfos = []      #FaceInformation of all frame in batch
        faceAligns = []     #Face Align of all frame in batch
        faceExpands = []     #Face crop of all frame in batch
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
                faceExpands.append(faceCropExpand)

        # print("Align Time:", time.time() - preTime)
        if len(faceAligns) > 0:
            preTime = time.time()
            descriptors = share_param.facerec_system.sdk.get_descriptor_batch(faceAligns)
            # preTime = time.time()
            # print(len(faceCropExpands))
            attributes = share_param.facerec_system.sdk.attributes.detect_batch(faceAligns)
            # print("Attributes of",len(faceAligns), time.time() - preTime)
            del faceAligns
            del faceExpands

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

            for faceInfo, attribute in zip(faceInfos, attributes):
                faceInfo.append(attribute)
        
        for deviceId, bbox, landmark, faceAlign, faceCropExpand, iBuffer, descriptor, user_name, score, attribute in faceInfos:
            faceFrameInfos[(iBuffer, deviceId)][0].append([bbox, landmark, faceAlign, faceCropExpand, descriptor, user_name, score, attribute])

        # preTime = time.time()
        share_param.tracking_multiCam.update(faceFrameInfos)
        # print("Tracking time:", time.time() - preTime)

        for iBuffer, deviceId in faceFrameInfos:
            faceInfos = faceFrameInfos[(iBuffer, deviceId)][0]
            rgb = faceFrameInfos[(iBuffer, deviceId)][1]
            rgbDraw = rgb.copy()

            for bbox, landmark, faceAlign, faceCropExpand, descriptor, user_name, score, attribute, trackid, overlap in faceInfos:
                faceCrop = rgb[int(bbox[1]):int(bbox[3]),
                                    int(bbox[0]):int(bbox[2])]

                faceSize = float((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
                isillumination, threshillumination = share_param.evaluter_cams[deviceID].check_illumination(faceCrop)
                isNotBlur, threshnotblur = share_param.evaluter_cams[deviceID].check_not_blur(faceCrop)
                isStraightFace = share_param.evaluter_cams[deviceID].check_straight_face(rgb, landmark)

                # if isNotBlur:
                #     new_user_name = datetime.now().strftime("%H%M%S%f") + ".jpg"
                #     photo_path = os.path.join("dataset/traindata", new_user_name)
                #     cv2.imwrite(photo_path, cv2.cvtColor(faceCropExpand, cv2.COLOR_RGB2BGR))
                # spamwriter.writerow([float(bbox[3]-bbox[1])*(bbox[2]-bbox[0]), float(threshnotblur), float(threshillumination)])
                
                if (deviceId,trackid) in trackidtoname:
                    if score > share_param.dev_config["DEV"]["face_reg_score"]:
                        trackidtoname[(deviceId,trackid)] = user_name
                        user_qualityscore_face_firsttime[user_name][6] = time.time()    #Update lastSeeTime
                        # print("1UpdateTime")

                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {} {:03.3f} {:03.3f} {:03.3f} {}".format(attribute, trackid, trackidtoname[(deviceId,trackid)], score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if isStraightFace and isillumination and not overlap and threshnotblur > user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][1]:
                        share_param.facerec_system.add_photo_descriptor_by_user_name(faceCropExpand, descriptor, trackidtoname[(deviceId,trackid)])
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][0] = faceSize
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][1] = threshnotblur
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][4] = faceCropExpand

                        # if user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][5] == True:
                        if score < share_param.dev_config["DEV"]["face_reg_score"]:
                            user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][5] = False
                        user_qualityscore_face_firsttime[trackidtoname[(deviceId,trackid)]][3] = time.time()
                        main_logger.info(f"Found a better face of {trackidtoname[(deviceId,trackid)]}")

                
                elif score > share_param.dev_config["DEV"]["face_reg_score"]:
                    trackidtoname[(deviceId,trackid)] = user_name
                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {} {:03.3f} {:03.3f} {:03.3f} {}".format(attribute, trackid, user_name, score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    # print("2UpdateTime")
                    user_qualityscore_face_firsttime[user_name][6] = time.time()    #Update lastSeeTime

                    if isStraightFace and isillumination and not overlap and threshnotblur > user_qualityscore_face_firsttime[user_name][1]:
                        share_param.facerec_system.add_photo_descriptor_by_user_name(faceCropExpand, descriptor, user_name)
                        user_qualityscore_face_firsttime[user_name][0] = faceSize
                        user_qualityscore_face_firsttime[user_name][1] = threshnotblur
                        user_qualityscore_face_firsttime[user_name][4] = faceCropExpand

                        # if user_qualityscore_face_firsttime[user_name][5] == True:
                        if score < share_param.dev_config["DEV"]["face_reg_score"]:
                            user_qualityscore_face_firsttime[user_name][5] = False
                        user_qualityscore_face_firsttime[user_name][3] = time.time()
                        main_logger.info(f"Found a better face of {user_name}")

                else:
                    new_user_name = datetime.now().strftime("%H%M%S%f")
                    cv2.rectangle(rgbDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                    cv2.putText(rgbDraw, "{} {} {} {:03.3f} {:03.3f} {:03.3f} {}".format(attribute,trackid, new_user_name, score, threshillumination, threshnotblur, overlap), (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    if isNotBlur and isStraightFace and isillumination and not overlap:
                        trackidtoname[(deviceId,trackid)] = new_user_name
                        share_param.facerec_system.add_photo_descriptor_by_user_name(faceCropExpand, descriptor, new_user_name)
                        user_qualityscore_face_firsttime[new_user_name] = [faceSize, threshnotblur, isStraightFace, time.time(), faceCropExpand, False, time.time(), "", 0]
                        filename = new_user_name + "F.jpg"
                        photo_path = os.path.join("dataset/firstphotos", filename)

                        first_face_bgr = cv2.cvtColor(faceCropExpand, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(photo_path, first_face_bgr)
                        support.add_redis_queue(new_user_name, first_face_bgr)

                        full_bgr =  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join("dataset/firstphotos", new_user_name + "U.jpg"), full_bgr)
                        
                        user_qualityscore_face_firsttime[new_user_name][4] = None
                        user_qualityscore_face_firsttime[new_user_name][5] = True
                        user_qualityscore_face_firsttime[new_user_name][8] += 1
                        main_logger.info(f"Add new face {new_user_name}")
            
            support.add_imshow_queue(deviceId, rgbDraw)

        # for deviceId in FPS:
        #     print(f"FPS {deviceId} {FPS[deviceId]}")
        
        main_logger.debug(f"Recogn Time: {time.time() - totalTime}")
    # csvfile.close()


def imshow_thread_fun():
    # writer = cv2.VideoWriter("appsrc ! videoconvert ! videoscale ! video/x-raw,width=320,height=240 ! theoraenc ! oggmux ! tcpserversink host=10.38.61.124 port=8080 recover-policy=keyframe sync-method=latest-keyframe unit-format=buffers units-max=1 buffers-max=0 sync=true ", 
                            # 0, 5, (320, 240), True)
             
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
                # writer.write(image)
            key = cv2.waitKey(10)

            if key == ord("q"):
                share_param.bExit = True
                main_logger.info(f"The shutdown command has been sent")


    # writer.release()
    cv2.destroyAllWindows()

def redis_thread_fun():
    global user_qualityscore_face_firsttime
    namespaces = {
        'ax233': 'http://entity.showroom.ewallet.lpb.com/xsd',
        'ax214': 'http://entity.ewallet.lpb.com/xsd',
        'ax216': "http://entity.showroom.ewallet.lpb.com/xsd"
    }
    namesays = {}

    while not share_param.bExit:
        if not share_param.bRunning:
            time.sleep(1)
        else:
            time.sleep(0.01)
        while not share_param.redis_queue.empty():
            user_name, image = share_param.redis_queue.get()
            base64_img = support.opencv_to_base64(image)
            soap_message = support.get_soap_message(base64_img)
            try:
                response = requests.post(share_param.dev_config["SOAP"]["url"], data = soap_message, headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction":""}, timeout=10)
            except requests.exceptions.ConnectTimeout as e:
                main_logger.error("faceSearch time out!")
                continue
            except:
                continue

            dom = ET.fromstring(response.content)
            customerNames = dom.findall(
                './/ax216:fullName',
                namespaces
            )

            scores = dom.findall(
                './/ax216:score',
                namespaces
            )
            vips = dom.findall(
                './/ax216:vip',
                namespaces
            )

            name_scores = []
            max_name = ""
            max_score = 0.0
            max_vip = "0"
            for name, score, vip in zip(customerNames, scores, vips):
                if name.text is None or score.text is None or vip.text is None:
                    continue
                try:
                    name_scores.append((name.text,float(score.text)))
                except:
                    print("Can't convert score string to number")
                    continue

                if float(score.text)>max_score:
                    max_score = float(score.text)
                    max_name = name.text
                    max_vip = vip.text

            print("max_name", max_name, "max_score", max_score, "max_vip", max_vip)

            if max_name is None or max_name == "" or max_score<0.85:
                continue
            
            if user_name in user_qualityscore_face_firsttime:
                user_qualityscore_face_firsttime[user_name][7] = max_name
            support.add_sayname_queue(max_name)

        try:
            while not share_param.sayname_queue.empty():
                name = share_param.sayname_queue.get()
                if name not in namesays:
                    namesays[name] = time.time()
                    support.say_name(name)
                else:
                    if time.time() - namesays[name] > 1800:
                        namesays[name] = time.time()
                        support.say_name(name)
                    else:
                        namesays[name] = time.time()
        except:
            print("Error can't say name")
            continue

if __name__ == '__main__':
    main_logger.info("Starting application")
    main_logger.info("Reading configs")
    share_param.sdk_config = support.get_config_yaml("configs/sdk_config.yaml")
    share_param.dev_config = support.get_config_yaml("configs/dev_config.yaml")
    share_param.cam_infos = support.get_config_yaml("configs/cam_infos.yaml")    
    main_logger.info("Done reading configs")
    share_param.batch_size = len(share_param.cam_infos["CamInfos"])
    main_logger.info("Init FaceRecognitionSystem")
    share_param.facerec_system = FaceRecognitionSystem(share_param.dev_config["DATA"]["photo_path"], share_param.sdk_config )
    main_logger.info("Init TrackingMultiCam")
    share_param.tracking_multiCam = TrackingMultiCam(share_param.sdk_config["tracking"])
    share_param.app = FaceRecogAPI()
    # main_logger.info("Init RedisClient")
    # share_param.redisClient = redis.StrictRedis(share_param.dev_config["REDIS"]["host"], share_param.dev_config["REDIS"]["port"])

    if share_param.dev_config["DATA"]["reload_database"]:
        share_param.facerec_system.create_database_from_folders(share_param.dev_config["DATA"]["photo_path"])
        share_param.facerec_system.save_database(share_param.dev_config["DATA"]["database_path"])
    share_param.facerec_system.load_database(share_param.dev_config["DATA"]["database_path"])

    share_param.cam_queue = queue.Queue(maxsize=share_param.CAM_QUEUE_SIZE*share_param.batch_size+1)
    share_param.detect_queue = queue.Queue(maxsize=share_param.DETECT_QUEUE_SIZE*share_param.batch_size+1)
    share_param.recogn_queue = queue.Queue(maxsize=share_param.RECOGN_QUEUE_SIZE*share_param.batch_size+1)
    share_param.imshow_queue = queue.Queue(maxsize=share_param.IMSHOW_QUEUE_SIZE*share_param.batch_size+1)
    share_param.redis_queue = queue.Queue(maxsize=share_param.REDIS_QUEUE_SIZE*share_param.batch_size+1)
    share_param.sayname_queue = queue.Queue(maxsize=share_param.SAYNAME_QUEUE_SIZE*share_param.batch_size+1)

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
    share_param.redis_thread = threading.Thread(target=redis_thread_fun, daemon=True, args=())
    share_param.api_thread = threading.Thread(target=uvicorn.run, daemon=True, 
                                            kwargs={"app": share_param.app, 
                                            "host": share_param.dev_config["APISERVER"]["host"], 
                                            "port": share_param.dev_config["APISERVER"]["port"]})

    for deviceID in share_param.cam_threads:
        share_param.cam_threads[deviceID].start()
    main_logger.info(f"cam_threads started")
    share_param.detect_thread.start()
    main_logger.info(f"detect_thread started")
    share_param.recogn_thread.start()
    main_logger.info(f"recogn_thread started")
    share_param.imshow_thread.start()
    main_logger.info(f"imshow_thread started")
    share_param.redis_thread.start()
    main_logger.info(f"redis_thread started")
    share_param.api_thread.start()
    main_logger.info(f"api_thread started")

    share_param.redis_thread.join()
    main_logger.info(f"redis_thread quited")
    share_param.imshow_thread.join()
    main_logger.info(f"imshow_thread quited")
    share_param.recogn_thread.join()
    main_logger.info(f"recogn_thread quited")
    share_param.detect_thread.join()
    main_logger.info(f"detect_thread quited")
    main_logger.info(f"Appication shutdown")