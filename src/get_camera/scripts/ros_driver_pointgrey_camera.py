#!/usr/bin/python3
import rospy
import EasyPySpin
import cv2
import numpy as np
from sensor_msgs.msg import Image
import PySpin
import time
from cv_bridge import CvBridge


def main():
    # ros
    rospy.init_node("pointgrey_publisher", anonymous=True)
    img_pub = rospy.Publisher('camera/rgb/image_pointgrey', Image, queue_size=2)
    bridge = CvBridge()

    # Instance creation
    cap = EasyPySpin.VideoCapture(0)
    cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)

    # Checking if it's connected to the camera
    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1

    # Set the camera parameters
    cap.set(cv2.CAP_PROP_EXPOSURE, 1000)  # -1 sets exposure_time to auto
    cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto
    cap.set(cv2.CAP_PROP_FPS, 25)

    image = Image()
    image.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image.encoding = "bgr8"

    # Start capturing
    while not rospy.is_shutdown():
        t0 = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)  # convert to BGR
            print("single frame time: %.3f s" % (time.time() - t0))

            # image.data = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).tobytes()
            # msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image.data = np.array(frame).tobytes()
            image.header.stamp = rospy.Time.now()
            img_pub.publish(image)

            img_show = cv2.resize(frame, None, fx=0.25, fy=0.25)
            cv2.imshow("press q to quit", img_show)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
