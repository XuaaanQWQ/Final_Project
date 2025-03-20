#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point  

class Orange:
    def __init__(self):
        rospy.init_node("debug_orange", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/realsense/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/realsense/depth/image_rect_raw", Image, self.depth_callback)
        self.center_pub = rospy.Publisher("/orange_center", Point, queue_size=10)
        self.latest_depth = None  

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            rospy.logerr("Depth image conversion error: %s", str(e))

    def get_depth(self, x, y):
        if self.latest_depth is None:
            return -1  
        h, w = self.latest_depth.shape
        if not (0 <= x < w and 0 <= y < h):
            return -1  
        depth_value = self.latest_depth[y, x]
        if depth_value <= 0 or np.isnan(depth_value):
           return -1  
        return depth_value  

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([15, 100, 100])
            upper_orange = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            orange_pixels = cv2.countNonZero(mask)

            if orange_pixels > 3:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 3:
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            depth = self.get_depth(cx, cy)

                            cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)  
                            cv2.drawContours(cv_image, [largest_contour], -1, (255, 0, 0), 2)  

                            point_msg = Point()
                            point_msg.x = cx
                            point_msg.y = cy
                            point_msg.z = depth  
                            self.center_pub.publish(point_msg)

                            rospy.loginfo(f"object:({cx}, {cy}), depth:{depth}")

            cv2.imshow("Orange", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Image processing error: %s", str(e))

if __name__ == "__main__":
    Orange()
    rospy.spin()
    cv2.destroyAllWindows()