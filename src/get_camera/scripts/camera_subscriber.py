#!/usr/bin/env python3
# !coding=utf-8

# right code !
# write by leo at 2018.04.26
# function:
# display the frame from another node.

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


def callback(data, args):
    print(float(data.header.stamp.secs) + float(data.header.stamp.nsecs) * 1e-9)
    cv_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    if args['save_img']:
        args['out_video'].write(cv_img)
    cv2.imshow("frame", cv_img)
    cv2.waitKey(3)


def displayWebcam():
    rospy.init_node('webcam_display', anonymous=True)
    opt = {
        # todo: 设置为True时保存视频，指定保存结果的路径
        'save_img': False,
        'dst_dir': "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/out.avi"
    }
    if opt['save_img']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1920, 1200)
        fps = 25
        out_video = cv2.VideoWriter(opt['dst_dir'], fourcc, fps, size)
    else:
        out_video = None
    opt['out_video'] = out_video
    rospy.Subscriber('camera/rgb/image_pointgrey', Image, callback, opt)
    rospy.spin()


if __name__ == '__main__':
    try:
        displayWebcam()
    except rospy.ROSInterruptException:
        pass
