#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Int32
import numpy as np
import cv2
from cv_bridge import CvBridge
import tf
import math
import torch
from ultralytics import YOLO
import os
import urllib.request

class DigitRegionCounter:
    def __init__(self):
        rospy.init_node("numm", anonymous=True)
        self.bridge = CvBridge()
        self.digit_map = None

        # Ensure YOLO model file exists or download it
        model_path = "yolov8n-digit.pt"
        if not os.path.exists(model_path):
            rospy.loginfo("模型不存在，正在下载 yolov8n-digit.pt ...")
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"  # 示例链接，需换成数字模型的地址
            try:
                urllib.request.urlretrieve(url, model_path)
                rospy.loginfo("模型下载完成！")
            except Exception as e:
                rospy.logerr("模型下载失败: %s", str(e))

        self.model = YOLO(model_path)

        # Subscribers using your camera topic names
        self.image_sub = rospy.Subscriber("/realsense/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/realsense/depth/image_rect_raw", Image, self.depth_callback)
        self.caminfo_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.caminfo_callback)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.ok_sub = rospy.Subscriber("/OK", Int32, self.ok_callback)

        # TF listener
        self.tf_listener = tf.TransformListener()

        # State
        self.map_data = None
        self.map_info = None
        self.K = None
        self.depth_image = None

    def caminfo_callback(self, msg):
        self.K = np.array(msg.K).reshape((3, 3))

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        if self.digit_map is None:
            self.digit_map = np.zeros((msg.info.height, msg.info.width), dtype=np.uint8)

    def image_callback(self, msg):
        if self.map_info is None or self.K is None or self.depth_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except:
            return

        results = self.model.predict(cv_image, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cx < 0 or cy < 0 or cy >= self.depth_image.shape[0] or cx >= self.depth_image.shape[1]:
                continue

            z = self.depth_image[cy, cx]
            if np.isnan(z) or z <= 0.2:
                continue

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx0, cy0 = self.K[0, 2], self.K[1, 2]
            X = (cx - cx0) * z / fx
            Y = (cy - cy0) * z / fy
            Z = z

            p_cam = PointStamped()
            p_cam.header.frame_id = "front_realsense_lens"
            p_cam.header.stamp = rospy.Time(0)
            p_cam.point.x = X
            p_cam.point.y = Y
            p_cam.point.z = Z

            try:
                self.tf_listener.waitForTransform("map", "front_realsense_lens", rospy.Time(0), rospy.Duration(1.0))
                p_map = self.tf_listener.transformPoint("map", p_cam)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                continue

            wx, wy = p_map.point.x, p_map.point.y
            mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)

            if 0 <= mx < self.map_info.width and 0 <= my < self.map_info.height:
                if self.map_data[my, mx] > 50:
                    self.digit_map[my, mx] = cls_id
                    rospy.loginfo("识别到数字 %d 在 map 像素 (%d, %d)", cls_id, mx, my)

    def ok_callback(self, msg):
        if msg.data == 1:
            self.analyze_digit_map()

    def analyze_digit_map(self):
        region_count = {}
        for d in range(10):
            mask = (self.digit_map == d).astype(np.uint8)
            if np.count_nonzero(mask) == 0:
                continue
            num_labels, _ = cv2.connectedComponents(mask)
            region_count[d] = num_labels - 1

        print("=== 连通区域分析结果 ===")
        for digit, count in region_count.items():
            print("数字 %d 出现了 %d 个区域" % (digit, count))

if __name__ == "__main__":
    DigitRegionCounter()
    rospy.spin()

