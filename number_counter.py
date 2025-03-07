import rospy
import cv2
import numpy as np
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from nav_msgs.msg import Odometry

# 初始化全局变量
found_numbers = {}  # 存储已识别的数字和数量 {number: count}
threshold_distance = 0.25  # 距离阈值，实际在改小一点
bridge = CvBridge()  
car_position = None  
car_orientation = None  
depth_image = None 

# 加载 YOLO 目标检测模型
model = YOLO("pretrained_model/yolov5x.pt")  # 如果慢就用m或者s
goal_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 目标数字类别

# 计算 3D 位置（从 bbox 获取相对坐标）
def get_3d_position(bbox, depth_image):
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_center = int((bbox[1] + bbox[3]) / 2)
    
    if depth_image is None:
        rospy.logwarn("深度图像未就绪！")
        return None

    depth = depth_image[y_center, x_center]

    # 假设摄像头的内参矩阵
    fx, fy, cx, cy = 800, 450, 640, 360 # csdn说的，待确定
    z = depth
    x = (x_center - cx) * z / fx
    y = (y_center - cy) * z / fy

    return (x, y, z)

# 坐标变换：将局部坐标转换为全局坐标
def transform_to_global(new_pos):
    if car_position is None or car_orientation is None:
        return None

    try:
        listener = tf.TransformListener()
        (trans, rot) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
        matrix = listener.fromTranslationRotation(trans, rot)
        relative_position = [new_pos[0], new_pos[1], new_pos[2], 1.0]
        global_position = np.dot(matrix, relative_position)[:3]  # 转换为全局坐标
        return tuple(global_position)  # 转换为元组存储
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr("坐标变换失败！")
        return None

# 判断是否为新检测的数字
def is_new_number(detected_number, global_position):
    """ 判断该数字是否已经存在于 found_numbers 中 """
    for existing_number, positions in found_numbers.items():
        for pos in positions:
            if existing_number == detected_number and np.linalg.norm(np.array(global_position) - np.array(pos)) < threshold_distance:
                return False  # 说明该数字已存在
    return True

# 记录数字并去重
def add_number_to_list(detected_number, global_position):
    if is_new_number(detected_number, global_position):
        if detected_number not in found_numbers:
            found_numbers[detected_number] = []
        found_numbers[detected_number].append(global_position)
        rospy.loginfo(f"检测到新的数字 {detected_number}, 当前位置: {global_position}")
        print(f"当前数字计数: {get_number_counts()}")

# 获取所有数字的出现次数
def get_number_counts():
    return {num: len(pos_list) for num, pos_list in found_numbers.items()}

# 获取出现最少的数字
def get_least_frequent_number():
    number_counts = get_number_counts()
    if not number_counts:
        return None
    return min(number_counts, key=number_counts.get)

# 目标检测
def detect_numbers(image):
    results = model(image) 
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = model.names[int(box.cls[0])]  # 获取类别名
            if class_id in goal_classes:  # 目标是数字
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                detections.append({"number": class_id, "bbox": bbox, "confidence": confidence})
    return detections

# 处理 RGB 图像
def rgb_image_callback(data):
    global depth_image
    try:
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"图像转换失败: {e}")
        return

    if depth_image is None:
        rospy.logwarn("深度图像未就绪！")
        return  

    # 运行 YOLO 目标检测
    detections = detect_numbers(rgb_image)

    # 遍历检测结果
    for detection in detections:
        if detection["confidence"] > 0.5:
            detected_number = detection["number"]  # 识别的数字
            bbox = detection["bbox"]
            relative_pos = get_3d_position(bbox, depth_image)
            if relative_pos:
                global_pos = transform_to_global(relative_pos)
                if global_pos:
                    add_number_to_list(detected_number, global_pos)

# 处理深度图像
def depth_image_callback(data):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(data, "32FC1")  # 32位浮点型深度图
    except CvBridgeError as e:
        rospy.logerr(f"深度图转换失败: {e}")

# 订阅里程计信息
def odometry_callback(msg):
    global car_position, car_orientation
    car_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
    car_orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                       msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

def main():
    rospy.init_node("number_tracking")

    # 订阅 RGB 图像
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_image_callback)
    # 订阅 深度图像
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_image_callback)
    # 订阅 里程计
    rospy.Subscriber("/odom", Odometry, odometry_callback)

    rospy.spin()
    
    # 获取出现最少的数字
    rospy.loginfo(f" the least number of occurrences is: {get_least_frequent_number()}")

if __name__ == "__main__":
    main()
