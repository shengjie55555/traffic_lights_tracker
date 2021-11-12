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

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from common import callback, load_param_gmsl2_3, load_param_pointgrey_3, detect_and_track_gmsl2_3, detect_and_track_pointgrey_3


def main():
    rospy.init_node("detector", anonymous=True)
    opt = load_param_pointgrey_3()
    
    rospy.Subscriber('Traffic_Lights_Num', Int8, callback, opt, queue_size=1)
    cap = cv2.VideoCapture("/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/test.avi")
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            if opt['flag']:
                detect_and_track_pointgrey_3(frame, opt)
            else:
                img0_show = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.namedWindow("press q to quit", 0)
                cv2.imshow("press q to quit", img0_show)
                if cv2.waitKey(10) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == "__main__":
    main()
    rospy.spin()