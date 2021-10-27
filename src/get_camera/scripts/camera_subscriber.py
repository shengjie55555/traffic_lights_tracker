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


def callback(data):
    # define picture to_down' coefficient of ratio
    print(float(data.header.stamp.secs) + float(data.header.stamp.nsecs) * 1e-9)
    scaling_factor = 1
    global bridge
    # global fourcc
    # global outfile
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    # outfile.write(cv_img)
    cv2.imshow("frame", cv_img)
    cv2.waitKey(3)


def displayWebcam():
    rospy.init_node('webcam_display', anonymous=True)

    # make a video_object and init the video object
    global bridge
    # global fourcc
    # global outfile
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # outfile = cv2.VideoWriter("/home/sheng/test.avi", fourcc, 30, (640, 480))
    bridge = CvBridge()
    rospy.Subscriber('camera/rgb/image_raw', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    displayWebcam()
