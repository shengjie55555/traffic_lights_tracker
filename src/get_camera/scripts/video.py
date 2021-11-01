#!/usr/bin/python3
import rospy
import EasyPySpin
import cv2
from sensor_msgs.msg import Image
import PySpin
import time
from cv_bridge import CvBridge


def main():
    # ros
    rospy.init_node("pt_grey_publisher", anonymous=True)
    img_pub = rospy.Publisher('camera/rgb/image_raw', Image, queue_size=2)
    bridge = CvBridge()

    # Instance creation
    cap = EasyPySpin.VideoCapture(0)
    cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)

    # Checking if it's connected to the camera
    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1

    # Set the camera parameters
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # -1 sets exposure_time to auto
    cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto
    cap.set(cv2.CAP_PROP_FPS, 25)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("/home/nvidia/users/wushengjie/wsj_ws/src/get_camera/data/test.avi", fourcc, fps, size)

    # Start capturing
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)  # convert to RGB
            print("single frame time: %.3f s" % (time.time() - t0))
            out.write(frame)

            # msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            # msg.header.stamp = rospy.Time.now()
            # img_pub.publish(msg)
            # print('** publishing webcam_frame ***')

            img_show = cv2.resize(frame, None, fx=0.25, fy=0.25)
            cv2.imshow("press q to quit", img_show)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()