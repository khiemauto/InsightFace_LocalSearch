import cv2
import numpy as np
from typing import List, Dict, Tuple
from core import share_param, support


TRACKER_INDEX     = 0
DESCRIPTOR_INDEX  = 1
UPDATED_INDEX     = 2

class Tracking():
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.trackers = {}  #trackid: [tracker, descriptor, updated]
        self.threshsimilarityinstant = config["threshsimilarityinstant"]
        self.threshiou = config["threshiou"]
        self.threshsimilarityiou = config["threshsimilarityiou"]
        self.maxid = 1  #Auto increament when creat new track
    
    def newsession(self, frame, detectboxs: List[int]):
        """
        box : x,y,w,h
        """

        trackboxs = {}

        for trackid in list(self.trackers):
            (success, trackbox) = self.trackers[trackid].update(frame)
            # if success:
            trackboxs[trackid] = trackbox
            # else:
            #     del self.trackers[trackid]

        track_detect_iou = {}   #(trackidmaxiou: detectboxid, detectbox, maxiou)
        detect_track_iou = {}   #(detectboxid: trackidmaxiou)

        for detectboxid, detectbox in enumerate(detectboxs):
            maxiou = 0
            trackidmaxiou = -1
            for trackid, trackbox in trackboxs.items():
                # print(detectbox, trackbox)
                xyxydetectbox = (detectbox[0], detectbox[1], detectbox[0]+detectbox[2], detectbox[1]+ detectbox[3])
                xyxytrackbox = (trackbox[0], trackbox[1], trackbox[0]+trackbox[2], trackbox[1]+ trackbox[3])
                iou = self.bb_intersection_over_union(xyxydetectbox, xyxytrackbox)
                if iou > maxiou:
                    maxiou = iou
                    trackidmaxiou = trackid
                    
            print(maxiou)
            if maxiou<self.threshiou:
                continue

            if trackidmaxiou not in track_detect_iou:
                track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou)
                detect_track_iou[detectboxid] = trackidmaxiou
            else:
                if maxiou > track_detect_iou[trackidmaxiou][2]:
                    track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou)
                    detect_track_iou[detectboxid] = trackidmaxiou
        
        trackid_bboxes = []
        #Clean tracker
        for trackid in list(self.trackers):
            if trackid in track_detect_iou:
                self.trackers[trackid] = cv2.legacy.TrackerKCF_create()
                self.trackers[trackid].init(frame, track_detect_iou[trackid][1])
                # self.trackers[trackid][1] = 0
                trackid_bboxes.append((trackid, track_detect_iou[trackid][1]))
            else:
                del self.trackers[trackid]
            
        #Create track with new detect
        for detectboxid, detectbox in enumerate(detectboxs):
            if detectboxid not in detect_track_iou:
                self.maxid += 1
                self.trackers[self.maxid] = cv2.legacy.TrackerKCF_create()
                self.trackers[self.maxid].init(frame, detectbox)
                trackid_bboxes.append((self.maxid, detectbox))

        # print(len(trackid_bboxes))
        return trackid_bboxes

    def update_with_descriptor(self, frame, detectbox_descriptors):
        """
        detectbox_descriptors : [(boxx, boxy, boxw, boxh), descriptor), ...]
        """

        #Create  track id return
        trackid_bboxes = [None]*len(detectbox_descriptors)

        for trackid in self.trackers:
            self.trackers[trackid][UPDATED_INDEX] = False

        #Matching descriptor
        detectbox_descriptor_processed = set()     #List of detectbox matched with trackid or created

        for trackid in self.trackers:
            track_descriptor = self.trackers[trackid][DESCRIPTOR_INDEX]
            self.trackers[trackid][UPDATED_INDEX] = False
            for detectboxid, (detectbox, descriptor) in enumerate(detectbox_descriptors):
                similarity = share_param.facerec_system.sdk.get_similarity(track_descriptor, descriptor)
                print("similarity:", similarity)
                if similarity > self.threshsimilarityinstant:
                    self.trackers[trackid][TRACKER_INDEX] = cv2.legacy.TrackerKCF_create()
                    self.trackers[trackid][TRACKER_INDEX].init(frame, detectbox)
                    self.trackers[trackid][DESCRIPTOR_INDEX] = descriptor
                    self.trackers[trackid][UPDATED_INDEX] = True
                    trackid_bboxes[detectboxid] = (trackid, detectbox)
                    detectbox_descriptor_processed.add(detectboxid)
                    break

        #Matching iou
        opencv_predict_trackboxs = {}

        for trackid in list(self.trackers):
            if self.trackers[trackid][UPDATED_INDEX]:
                continue
            (success, trackbox) = self.trackers[trackid][TRACKER_INDEX].update(frame)
            if success:
                opencv_predict_trackboxs[trackid] = trackbox
            else:
                del self.trackers[trackid]
        
        # print("opencv_predict_trackboxs",opencv_predict_trackboxs)

        track_detect_iou = {}   #(trackidmaxiou: detectboxid, detectbox, maxiou, descriptor)
        # detect_track_iou = {}   #(detectboxid: trackidmaxiou)

        for detectboxid, (detectbox, descriptor) in enumerate(detectbox_descriptors):
            #Skip detectbox_descriptor processed
            if detectboxid in detectbox_descriptor_processed:
                continue

            maxiou = 0.0
            trackidmaxiou = -1
            for opencv_trackid in opencv_predict_trackboxs:
                opencv_trackbox = opencv_predict_trackboxs[opencv_trackid]
                # print(detectbox, trackbox)
                xyxydetectbox = (detectbox[0], detectbox[1], detectbox[0]+detectbox[2], detectbox[1]+ detectbox[3])
                xyxytrackbox = (opencv_trackbox[0], opencv_trackbox[1], opencv_trackbox[0]+opencv_trackbox[2], opencv_trackbox[1]+ opencv_trackbox[3])
                iou = self.bb_intersection_over_union(xyxydetectbox, xyxytrackbox)
                track_descriptor = self.trackers[opencv_trackid][DESCRIPTOR_INDEX]
                similarity = share_param.facerec_system.sdk.get_similarity(track_descriptor, descriptor)
                print("iou", iou, "similarity", similarity)
                # if iou > maxiou:
                if iou > maxiou and similarity>self.threshsimilarityiou:
                    maxiou = iou
                    trackidmaxiou = opencv_trackid

            # print(maxiou, maxiou)
            if trackidmaxiou == -1 or maxiou<self.threshiou:
                continue

            if trackidmaxiou not in track_detect_iou:
                track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou, descriptor)
                # detect_track_iou[detectboxid] = trackidmaxiou
            else:
                if maxiou > track_detect_iou[trackidmaxiou][2]:
                    track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou, descriptor)

                    #Remove detect_track_iou has smaller maxiou
                    # for key, value in detect_track_iou.items():
                    #     if value == trackidmaxiou:
                    #         del detect_track_iou[key]
                    #         break

                    # detect_track_iou[detectboxid] = trackidmaxiou

        
        
        #Clean tracker
        for trackid in list(self.trackers):
            if self.trackers[trackid][UPDATED_INDEX]:
                continue
            if trackid in track_detect_iou:
                self.trackers[trackid][TRACKER_INDEX] = cv2.legacy.TrackerKCF_create()
                self.trackers[trackid][TRACKER_INDEX].init(frame, track_detect_iou[trackid][1])
                self.trackers[trackid][DESCRIPTOR_INDEX] = track_detect_iou[trackid][3]
                self.trackers[trackid][UPDATED_INDEX] = True
                trackid_bboxes[track_detect_iou[trackid][0]] = (trackid, track_detect_iou[trackid][1])
                detectbox_descriptor_processed.add(track_detect_iou[trackid][0])
            else:
                del self.trackers[trackid]
            
        #Create track with new detect

        # print("detect_track_iou", detect_track_iou)

        overlaps = [False]*len(detectbox_descriptors)

        for detectboxid, (detectbox,descriptor) in enumerate(detectbox_descriptors):
            #Check over lap
            xyxydetectbox = (detectbox[0], detectbox[1], detectbox[0]+detectbox[2], detectbox[1]+ detectbox[3])

            for other_detectboxid, (other_detectbox,other_descriptor) in enumerate(detectbox_descriptors):
                if detectboxid == other_detectboxid:
                    continue
                other_xyxydetectbox = (other_detectbox[0], other_detectbox[1], other_detectbox[0]+other_detectbox[2], other_detectbox[1]+ other_detectbox[3])
                iou = self.bb_intersection_over_union(xyxydetectbox, other_xyxydetectbox)
                if iou > 0.01:
                    overlaps[detectboxid] = True
                    break
            #New track
            if detectboxid in detectbox_descriptor_processed:
                continue
            # if detectboxid not in detect_track_iou:
            self.maxid += 1
            self.trackers[self.maxid] = [cv2.legacy.TrackerKCF_create(), descriptor, True]
            self.trackers[self.maxid][0].init(frame, detectbox)
            trackid_bboxes[detectboxid] = (self.maxid, detectbox)
            detectbox_descriptor_processed.add(detectboxid)

        # print(len(trackid_bboxes))
        return trackid_bboxes, overlaps

    def release(self):
        self.trackers.clear()

    def update(self, frame):
        ret = []
        for trackid in list(self.trackers):
            (success, boxes) = self.trackers[trackid].update(frame)
            # if success:
            ret.append((trackid, boxes))
            # else:
                # del self.trackers[trackid]
        return ret
        
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

class TrackingMultiCam():
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.trackers: Dict[int:Tracking] = {}
    
    def update(self, faceFrameInfos):
        """
        faceFrameInfos: {(iBuffer, deviceId): [faceinfos, rgb],}
        faceinfos : [[bbox, landmark, faceCropExpand, descriptor, user_name, score],]
        rgb : rgb image
        """
        

        for iBuffer, deviceId in faceFrameInfos:
            if deviceId not in self.trackers:
                self.trackers[deviceId] = Tracking(self.config)

            faceInfos = faceFrameInfos[(iBuffer, deviceId)][0]
            rgb = faceFrameInfos[(iBuffer, deviceId)][1]

            detectbox_descriptors = []
            descriptors = []

            for bbox, landmark, faceAlign, faceCropExpand, descriptor, user_name, score, attribute in faceInfos:
                x_l, y_t, x_r, y_b, conf = bbox 
                boxx, boxy, boxw, boxh = x_l, y_t, x_r-x_l, y_b-y_t

                detectbox_descriptors.append(((boxx, boxy, boxw, boxh), descriptor))

            trackinfos, overlaps = self.trackers[deviceId].update_with_descriptor(rgb, detectbox_descriptors)

            for faceInfo, (trackid, boxes), overlap in zip(faceInfos, trackinfos, overlaps):
                faceInfo.append(trackid)
                faceInfo.append(overlap)
