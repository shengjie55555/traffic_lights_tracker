#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import sys
 
 
def webcamImagePub():
    rospy.init_node('webcam_puber', anonymous=True)
    img_pub = rospy.Publisher('camera/rgb/image_raw', Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(30)
 
    cap = cv2.VideoCapture("/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/test.avi")
 
    if not cap.isOpened():
        sys.stdout.write("Webcam is not available !")
        return -1
 
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            img_pub.publish(msg)
            print('** publishing webcam_frame ***')
        else:
            rospy.loginfo("Capturing image failed.")
        rate.sleep()
 
if __name__ == '__main__':
    try:
        webcamImagePub()
    except rospy.ROSInterruptException:
        pass
