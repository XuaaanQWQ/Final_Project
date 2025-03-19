#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class OrangeDebugger:
    def __init__(self):
        rospy.init_node("debug_orange", anonymous=True)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/front/image_raw", Image, self.image_callback)

        self.center_pub = rospy.Publisher("/orange_center", Point, queue_size=10)

    def image_callback(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            lower_orange = np.array([15, 100, 100])
            upper_orange = np.array([25, 255, 255])

            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            orange_pixels = cv2.countNonZero(mask)

            if orange_pixels > 20:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:

                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 20:

                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            point_msg = Point()
                            point_msg.x = cx
                            point_msg.y = cy
                            self.center_pub.publish(point_msg)

                            print(f"orange: ({cx}, {cy})")

                            cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.drawContours(cv_image, [largest_contour], -1, (0, 0, 255), 2)
            
            cv2.imshow("123", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Image processing error: %s", str(e))

if __name__ == "__main__":
    OrangeDebugger()
    rospy.spin()
    cv2.destroyAllWindows()