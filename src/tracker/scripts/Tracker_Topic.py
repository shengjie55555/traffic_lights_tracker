#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/25 下午5:01
# @Author :
# @File : Tracker.py
# @Software: CLion

from os import name

from numpy.core.fromnumeric import argmin, mean
from numpy.core.numeric import ones
import rospy
from tracker.msg import traffic_lights_num, traffic_lights_state
from sensor_msgs.msg import Image

import time
import cv2
import torch
import numpy as np
import copy

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from Tracker_Local import Traffic_Light_Filter, detect_and_track, xyxy_to_xywh, draw_boxes, draw_results, get_patch, transform, callback


def image_callback(data, args):
    args["image"] = data
    args["image_flag"] = True


def main():
    rospy.init_node("detector", anonymous=True)
    opt = {
        "weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/runs/exp1/weights/best.pt",
        "deep_sort_weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
        "config_deepsort": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/configs/deep_sort.yaml",
        "imgsz": 640,
        "augment": False,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "patch_mode": True,
        "agnostic_nms": False,
        "classes": None,
        "view_img": True,
        "save_img": False,
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

    # Initial deepsort
    cfg = get_config()
    cfg.merge_from_file(opt["config_deepsort"])
    deepsort = DeepSort(opt["deep_sort_weights"],
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Get names and colors: ['vehicle', 'pedestrian', 'red', 'green', 'sign_p', 'sign_w', 'sign_m', 'light']
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[139, 139, 0], [0, 165, 255],
              [0, 0, 255], [0, 255, 0],
              [180, 105, 255], [0, 255, 255], [255, 0, 0],
              [128, 128, 128]]

    # Save result as video
    if opt['save_img']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1920, 1200)
        fps = 25
        out_video = cv2.VideoWriter("/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/out.avi", fourcc, fps, size)
    else:
        out_video = None

    temp = {
        "device": device,
        "half": half,
        "model": model,
        "deepsort_model": deepsort,
        "imgsz": imgsz,
        "names": names,
        "colors": colors,
        "out_video": out_video,
        "flag": False,
        "filter": Traffic_Light_Filter(num=3, maxsize=5, init=True, init_data=2),
        "lights_num": 0,
        "image": Image(),
        "image_flag": False,
        "pub": rospy.Publisher("Traffic_Lights_State", traffic_lights_state, queue_size=1),
        "state_msg": traffic_lights_state()
    }

    opt.update(temp)

    rospy.Subscriber('Traffic_Lights_Num', traffic_lights_num, callback, opt, queue_size=1)
    rospy.Subscriber('camera/rgb/image_raw', Image, image_callback, opt, queue_size=1)
    
    while not rospy.is_shutdown():
        if opt["image_flag"]:
            opt["image_flag"] = False
            frame = np.frombuffer(opt["image"].data, dtype=np.uint8).reshape(opt["image"].height, opt["image"].width, -1)
            if opt['flag']:
                detect_and_track(frame, opt)
            else:
                img0_show = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.namedWindow("press q to quit", 0)
                cv2.imshow("press q to quit", img0_show)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == "__main__":
    main()
    rospy.spin()
