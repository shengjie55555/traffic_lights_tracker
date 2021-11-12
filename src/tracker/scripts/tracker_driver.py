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
from std_msgs.msg import Int8

import time
import cv2
import torch
import numpy as np
import copy

import EasyPySpin
import PySpin

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from common import callback, load_param_pointgrey_3, load_param_pointgrey_8, detect_and_track_pointgrey_3, detect_and_track_pointgrey_8


def main():
    rospy.init_node("detector", anonymous=True)
    opt = load_param_pointgrey_8

    rospy.Subscriber('Traffic_Lights_Num', Int8, callback, opt, queue_size=1)
    
    # Instance creation
    cap = EasyPySpin.VideoCapture(0)
    cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)

    # Checking if it's connected to the camera
    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1

    # Set the camera parameters
    cap.set(cv2.CAP_PROP_EXPOSURE, 1000)  # -1 sets exposure_time to auto, us
    cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto
    cap.set(cv2.CAP_PROP_FPS, 25)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            if opt['flag']:
                detect_and_track_pointgrey_8(frame, opt)
            else:
                img0_show = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.namedWindow("press q to quit", 0)
                cv2.imshow("press q to quit", img0_show)
                if cv2.waitKey(10) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == "__main__":
    main()
    rospy.spin()
