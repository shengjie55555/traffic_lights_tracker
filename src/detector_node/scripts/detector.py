#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 上午8:47
# @Author :
# @File : detector.py
# @Software: CLion

import rospy
from sensor_msgs.msg import Image

import time
import cv2
import torch
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox
from detector_node.msg import target, multi_target


def callback(data, args):
    with torch.no_grad():
        multi_t = multi_target()
        # Run inference
        t0 = time.time()

        # Read img from sensor_msg
        img0 = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        img0 = np.ascontiguousarray(img0)

        # Padded resize
        img = letterbox(img0, new_shape=args["imgsz"])[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(args["device"])
        img = img.half() if args["half"] else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = args["model"](img, augment=args["augment"])[0]

        # Apply NMS
        pred = non_max_suppression(pred, args["conf_thres"], args["iou_thres"],
                                   classes=args["classes"], agnostic=args["agnostic_nms"])
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, args["names"][int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # publish results
                        single_t = target()
                        single_t.Class = args["names"][int(cls)]
                        single_t.score = conf
                        single_t.xmin = int(xyxy[0])
                        single_t.ymin = int(xyxy[1])
                        single_t.xmax = int(xyxy[2])
                        single_t.ymax = int(xyxy[3])
                        multi_t.targets.append(single_t)

                        if args["view_img"]:  # Add bbox to image
                            label = '%s %.2f' % (args["names"][int(cls)], conf)
                            plot_one_box(xyxy, img0, label=label, color=args["colors"][int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if args["view_img"]:
                    cv2.namedWindow("detect_result", 0)
                    cv2.imshow("detect_result", img0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
        
        if pred is None:
            cv2.namedWindow("original_image", 0)
            cv2.imshow("original_image", img0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        
        args["pub"].publish(multi_t)

        print('Done. (%.3fs)' % (time.time() - t0))


def main():
    rospy.init_node("detector", anonymous=True)
    opt = {
        # todo: 修改权重的实际位置
        "weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/detector_node/scripts/best.pt",
        "imgsz": 640,
        "augment": False,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "agnostic_nms": False,
        "classes": None,
        "view_img": True,
        "device": ''}

    # Initialize
    set_logging()
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt["weights"], map_location=device)  # load FP32 model
    imgsz = check_img_size(opt["imgsz"], s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    temp = {
        "device": device,
        "half": half,
        "model": model,
        "imgsz": imgsz,
        "names": names,
        "colors": colors,
        "pub": rospy.Publisher("/bboxes", multi_target, queue_size=1)
    }

    opt.update(temp)

    rospy.Subscriber('camera/rgb/image_pointgrey', Image, callback, opt, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
