#!/usr/bin/env python
"""
 Copyright (c) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from numpy.core.fromnumeric import resize
from openvino.inference_engine import IECore
import openvino
# import ngraph as ng

import torch
from insight_face.modules.detection.retinaface.model_class import RetinaFace
from insight_face.modules.alignment.align_faces import align_and_crop_face
import cv2
from core import support
import time
sdk_config = support.get_config_yaml("configs/sdk_config.yaml")
openvino_config = support.get_config_yaml("configs/sdk_config_openvino.yaml")

detector = RetinaFace(sdk_config["detector"])


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Loading Inference Engine")

    detector_config = openvino_config["detector"]
    detector_arch = detector_config["architecture"]
    detector_arch_config = openvino_config[detector_arch]

    print(detector_arch_config["output_names"])
    ie = IECore()
    # ie.set_config({"DYN_BATCH_ENABLED": "YES"})
    # ---1. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format ---
    log.info(f'Loading network: {detector_arch_config["weights_path"]}')
    # IENetwork(xml, bin)
    detector_net = ie.read_network(detector_arch_config["weights_path"] + ".xml", detector_arch_config["weights_path"] + ".bin")

    embedder_net = ie.read_network("weights/iresnet34.xml", "weights/iresnet34.bin")
    #---------------------------------2. Set batch size ---------------------------------------------------
    detector_net.batch_size = 10
    embedder_net.batch_size = 10

    # ie.set_config(config={"DYN_BATCH_ENABLED": "YES"}, device_name=openvino_config['detector']['device'])
    # batch = 1
    # shapes = {}
    # for input_layer in net.input_info:
    #     new_shape = [batch] + net.input_info[input_layer].input_data.shape[1:]
    #     shapes.update({input_layer: new_shape})

    # log.info(f"Set batch size: {batch}")
    # net.reshape(shapes)
    # --------------------------- 3. Read and preprocess input --------------------------------------------
    detector_input_name = ""
    for input_key in detector_net.input_info:
        if len(detector_net.input_info[input_key].input_data.layout) == 4:
            detector_n, detector_c, detector_h, detector_w = detector_net.input_info[input_key].input_data.shape
            detector_input_name = input_key
            # detector_net.input_info[input_key].precision = 'I8'
    print(detector_n, detector_c, detector_h, detector_w)

    embedder_input_name = ""
    for input_key in embedder_net.input_info:
        if len(embedder_net.input_info[input_key].input_data.layout) == 4:
            embedder_n, embedder_c, embedder_h, embedder_w = embedder_net.input_info[input_key].input_data.shape
            embedder_input_name = input_key
    print(embedder_n, embedder_c, embedder_h, embedder_w)

    # print(n,c,h,w)

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # log.info('Preparing output blobs')
    output_names = embedder_net.outputs.keys()
    print(output_names)
    # output_infos = net.outputs.values()

    # if len(output_names) == 0:
    #     log.error("Can't find a ouput layer in the topology")
    # output_dims = []
    
    # for output_info in output_infos:
    #     output_dims.append(output_info.shape)

    # if len(output_dims) != 3:
    #     log.error("Incorrect output dimensions for Retina model")

    # for output_info in detector_net.outputs.values():
    #     output_info.precision = "FP16"
    # for output_info in embedder_net.outputs.values():
    #     output_info.precision = "FP16"
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    # exec_net = ie.load_network(net, 'CPU', {"DYN_BATCH_ENABLED": "YES"})
    detector_exec_net = ie.load_network(network=detector_net, device_name=openvino_config['detector']['device'])
    embedder_exec_net = ie.load_network(network=embedder_net, device_name=openvino_config['detector']['device'], config={"DYN_BATCH_ENABLED": "YES"})
    
    print(type(embedder_exec_net))
    cap = cv2.VideoCapture(0)
    FPS = [20.0, time.time(), 0]

    data = {}

    for i in range(10000):
        if time.time()-FPS[1] < 1.0:
            FPS[2] += 1
        else:
            FPS[0] = 0.8*FPS[0] + 0.2*FPS[2]
            FPS[2] = 0
            FPS[1] = time.time()
            print(f"Pipleline FPS : {FPS[0]}")


        # image = self._preprocess(image)
        # self.model_input_shape = image.shape
        # raw_pred = self._predict_raw(image)
        # bboxes, landms = self._postprocess(raw_pred)


        raw_images = []
        images = np.ndarray(shape=(detector_n, detector_c, detector_h, detector_w ))

        for i in range(detector_n):

            grabed, frame= cap.read()
            if not grabed:
                continue

            xstart = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:, xstart: xstart + frame.shape[0]]
            raw_images.append(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = detector._preprocess(rgb)

            print(rgb.shape)
            detector.model_input_shape = rgb.shape
            rgb = rgb.transpose((2, 0, 1))

            images[i] = rgb

   
        # data[detector_input_name] = images
        preTime = time.time()
        res = detector_exec_net.infer(inputs={detector_input_name:images})
        print("Detect:", time.time()-preTime)

        # print(res["515"].shape)
        for image, re0, re1, re2 in zip(raw_images , res[detector_arch_config["output_names"][0]],res[detector_arch_config["output_names"][1]], res[detector_arch_config["output_names"][2]]):
            draw_image = image.copy()
    
            raw_pred = (torch.from_numpy(re0),torch.from_numpy(re1),torch.from_numpy(re2))

            bboxes, landms = detector._postprocess(raw_pred)
            converted_landmarks = []
            # convert to our landmark format (2,5)
            for landmarks_set in landms:
                x_landmarks = []
                y_landmarks = []
                for i, lm in enumerate(landmarks_set):
                    if i % 2 == 0:
                        x_landmarks.append(lm)
                    else:
                        y_landmarks.append(lm)
                converted_landmarks.append(x_landmarks + y_landmarks)

            landmarks = np.array(converted_landmarks)

            if len(bboxes) == 0:
                continue

            faces = np.ndarray(shape=(10, 3, 112, 112 ))

            for i, (bbox, landmark) in enumerate(zip( bboxes, landmarks)):
                
                # draw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                face = align_and_crop_face(image, landmark, size=(112, 112))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                face = face.astype(np.float32)/255.0

                # print(face.shape)
                mean = [0.5] * 3
                std = [0.5 * 256 / 255] * 3
                face = (face-mean)/std

                face = face.transpose((2, 0, 1))
                faces[i] = face

                cv2.rectangle(draw_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                cv2.putText(draw_image, f"{bbox[4]}", (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # print(landmark)
                # landmark = landmark.reshape((5,2), order='F')
                # for pts in landmark:
                    # cv2.drawMarker(draw_image, (int(pts[0]), int(pts[1])), (0,255,0), cv2.MARKER_CROSS, markerSize=5, thickness=2)
                # print(landmark)
            
            # _preprocess

            # inputs_count = 2
            # if 
            embedder_exec_net.requests[0].set_batch(len(bboxes))
            preTime = time.time()
            # embedder_exec_net.requests[0].inputs[embedder_input_name] = faces
            embedder_exec_net.requests[0].infer(inputs= {embedder_input_name: faces})
            res = embedder_exec_net.requests[0].outputs['475']
            # res = res['475']
            print(res.shape)

            for i in range(10):
                res[i] = res[i] / np.linalg.norm(res[i])
                print(res[i].sum())
            # print(res['475'][1])
            print("Embedd:", time.time()-preTime)
                
            cv2.imshow("RESULT", draw_image)
        cv2.waitKey(10)


if __name__ == '__main__':
    sys.exit(main() or 0)
