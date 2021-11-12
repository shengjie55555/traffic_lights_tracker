#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import  numpy as np
import cv2
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    rospy.init_node("gmsl2_camera_publisher", anonymous=True)
    img_pub = rospy.Publisher('camera/rgb/image_gmsl2', Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(30)

    cap = cv2.VideoCapture(0)

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


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
