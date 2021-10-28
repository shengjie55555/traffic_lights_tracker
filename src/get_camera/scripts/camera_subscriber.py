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
    print(float(data.header.stamp.secs) + float(data.header.stamp.nsecs) * 1e-9)
    cv_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    cv2.imshow("frame", cv_img)
    cv2.waitKey(3)


def displayWebcam():
    rospy.init_node('webcam_display', anonymous=True)
    rospy.Subscriber('camera/rgb/image_raw', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        displayWebcam()
    except rospy.ROSInterruptException:
        pass
