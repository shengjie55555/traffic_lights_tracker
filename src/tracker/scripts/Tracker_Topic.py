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

from common import load_param_8, detect_and_track_8, callback, load_param_2, detect_and_track_2


def image_callback(data, args):
    args["image"] = data
    args["image_flag"] = True


def main():
    rospy.init_node("detector", anonymous=True)
    opt = load_param_8()
    opt["image_flag"] = False
    opt["image"] = Image()

    rospy.Subscriber('Traffic_Lights_Num', traffic_lights_num, callback, opt, queue_size=1)
    rospy.Subscriber('camera/rgb/image_raw', Image, image_callback, opt, queue_size=1)
    
    while not rospy.is_shutdown():
        if opt["image_flag"]:
            frame = np.frombuffer(opt["image"].data, dtype=np.uint8).reshape(opt["image"].height, opt["image"].width, -1)
            if opt['flag']:
                detect_and_track_8(frame, opt)
            else:
                img0_show = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.namedWindow("press q to quit", 0)
                cv2.imshow("press q to quit", img0_show)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == "__main__":
    main()
    rospy.spin()
