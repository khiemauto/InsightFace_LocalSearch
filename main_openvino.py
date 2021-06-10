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
from openvino.inference_engine import IECore
import openvino
import ngraph as ng

import torch
from insight_face.modules.detection.retinaface.model_class import RetinaFace
import cv2
from core import support
import time
sdk_config = support.get_config_yaml("configs/sdk_config.yaml")

detector = RetinaFace(sdk_config["detector"])

# def build_argparser():
#     parser = ArgumentParser(add_help=False)
#     args = parser.add_argument_group("Options")
#     args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
#     args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
#                       required=True, type=str)
#     args.add_argument("-i", "--input", help="Required. Path to an image file.",
#                       required=True, type=str)
#     args.add_argument("-l", "--cpu_extension",
#                       help="Optional. Required for CPU custom layers. "
#                            "Absolute path to a shared library with the kernels implementations.",
#                       type=str, default=None)
#     args.add_argument("-d", "--device",
#                       help="Optional. Specify the target device to infer on; "
#                            "CPU, GPU, FPGA or MYRIAD is acceptable. "
#                            "Sample will look for a suitable plugin for device specified (CPU by default)",
#                       default="CPU", type=str)
#     args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
#     args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

#     return parser

args = {
    "model":"weights/res50",
    "input":"dataset/bestphotos/113900549523G.jpg",
    "device":"CPU",
    'cpu_extension': False,
    "labels":"",
    "nt":""
}

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Loading Inference Engine")
    ie = IECore()
    
    # ---1. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format ---
    model_path = args["model"]
    log.info(f"Loading network:\n\t{model_path}")
    net = ie.read_network(model_path+ ".xml", model_path + ".bin")
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Device info:")
    versions = ie.get_versions(args['device'])
    print(f"{' ' * 8}{args['device']}")
    print(f"{' ' * 8}MKLDNNPlugin version ......... {versions[args['device']].major}.{versions[args['device']].minor}")
    print(f"{' ' * 8}Build ........... {versions[args['device']].build_number}")

    if args['cpu_extension'] and "CPU" in args['device']:
        ie.add_extension(args['cpu_extension'], "CPU")
        log.info(f"CPU extension loaded: {args['cpu_extension']}")
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------
    for input_key in net.input_info:
        print(input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape

    # print(n,c, h,w)

    images = np.ndarray(shape=(n, c, h, w))
    print(images.shape)
    images_hw = []
    for i in range(n):

        image = cv2.imread("/mnt/LINUXDATA/Source/.data/face_attributes/01000-20210525T030241Z-001/01000/01022.png")
        image = cv2.resize(image, (224,224))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rgb = detector._preprocess(rgb)
        detector.model_input_shape = rgb.shape
        rgb = rgb.transpose((2, 0, 1))

        print(rgb.shape)
        images[i] = rgb

        # image = cv2.imread(args['input'])
        # ih, iw = image.shape[:-1]
        # images_hw.append((ih, iw))
        # log.info("File was added: ")
        # log.info(f"        {args['input']}")
        # if (ih, iw) != (h, w):
        #     log.warning(f"Image {args['input']} is resized from {image.shape[:-1]} to {(h, w)}")
        #     image = cv2.resize(image, (w, h))
        # image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        # images[i] = image
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info("Preparing input blobs")
    assert (len(net.input_info.keys()) == 1 or len(
        net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""

    for input_key in net.input_info:
        if len(net.input_info[input_key].layout) == 4:
            input_name = input_key
            net.input_info[input_key].precision = 'U8'
        elif len(net.input_info[input_key].layout) == 2:
            input_info_name = input_key
            net.input_info[input_key].precision = 'FP32'
            if net.input_info[input_key].input_data.shape[1] != 3 and net.input_info[input_key].input_data.shape[1] != 6 or \
                net.input_info[input_key].input_data.shape[0] != 1:
                log.error('Invalid input info. Should be 3 or 6 values length.')

    data = {}
    data[input_name] = images

    if input_info_name != "":
        detection_size = net.input_info[input_info_name].input_data.shape[1]
        infos = np.ndarray(shape=(n, detection_size), dtype=float)
        for i in range(n):
            infos[i, 0] = h
            infos[i, 1] = w
            for j in range(2, detection_size):
                infos[i, j] = 1.0
        data[input_info_name] = infos

    print(input_name)

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    # output_name, output_info = "", None

    output_names = []
    output_infos = []

    func = ng.function_from_cnn(net)

    # print(net.outputs)
    if func:
        ops = func.get_ordered_ops()
        for op in ops:
            if op.friendly_name in net.outputs:
                # print(op)
                output_names.append(op.friendly_name)
                output_infos.append(net.outputs[op.friendly_name])
                # break
    else:
        output_name = list(net.outputs.keys())[0]
        output_info = net.outputs[output_name]

    # print(output_names)
    # print(output_infos)

    if len(output_names) == 0:
        log.error("Can't find a DetectionOutput layer in the topology")
    # print(output_info)
    output_dims = []
    
    for output_info in output_infos:
        output_dims.append(output_info.shape)

    print(output_dims)
    if len(output_dims) != 3:
        log.error("Incorrect output dimensions for Retina model")

    for output_info in output_infos:
        output_info.precision = "FP16"
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=args['device'])
    log.info("Creating infer request and starting inference")
    
    cap = cv2.VideoCapture(0)
    for i in range(10000):
        images = np.ndarray(shape=(n, c, h, w))
        for i in range(n):

            grabed, frame= cap.read()
            if not grabed:
                continue

            xstart = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:, xstart: xstart + frame.shape[0]]
            image = cv2.resize(frame, (224,224))

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            rgb = detector._preprocess(rgb)
            detector.model_input_shape = rgb.shape
            rgb = rgb.transpose((2, 0, 1))

            images[i] = rgb

        # data = {}
        data[input_name] = images

        # if input_info_name != "":
        #     detection_size = net.input_info[input_info_name].input_data.shape[1]
        #     infos = np.ndarray(shape=(n, detection_size), dtype=float)
        #     for i in range(n):
        #         infos[i, 0] = h
        #         infos[i, 1] = w
        #         for j in range(2, detection_size):
        #             infos[i, j] = 1.0
        #     data[input_info_name] = infos

        preTime = time.time()
        res = exec_net.infer(inputs=data)
        print("Detect:", time.time()-preTime)
        raw_pred = (torch.from_numpy(res["762"]),torch.from_numpy(res["837"]),torch.from_numpy(res["836"]))

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

        for bbox, landmark in zip( bboxes, landmarks):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
            cv2.putText(image, f"{bbox[4]}", (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            

        cv2.imshow("RESULT", image)
        cv2.waitKey(10)


if __name__ == '__main__':
    sys.exit(main() or 0)
