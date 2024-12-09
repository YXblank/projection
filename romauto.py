import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TransparentObjectExtractor:
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = None

        # 订阅图像话题
        rospy.Subscriber('/kinect2/camera/hd/image_color', Image, self.image_callback)

        # 设置定时器，每10秒调用一次区域生长算法
        self.timer = rospy.Timer(rospy.Duration(10), self.run_region_growth)

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Error converting image: %s", str(e))

    def extract_transparent_objects(self, image, lower_hsv, upper_hsv):
        # 将图像转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建掩模
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # 清理掩模（可选）
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # 提取透明对象
        transparent_objects = cv2.bitwise_and(image, image, mask=cleaned_mask)

        return transparent_objects, cleaned_mask

    def run_region_growth(self, event):
        if self.cv_image is not None:
            # 定义HSV颜色范围（根据需要调整）
            lower_hsv = np.array([0, 0, 200])  # 示例值
            upper_hsv = np.array([180, 25, 255])  # 示例值

            # 提取透明对象
            transparent_objects, cleaned_mask = self.extract_transparent_objects(self.cv_image, lower_hsv, upper_hsv)

            # 显示结果
            cv2.imshow("Transparent Objects", transparent_objects)
            cv2.imshow("Mask", cleaned_mask)
            cv2.waitKey(1)

def main():
    rospy.init_node('transparent_object_extractor', anonymous=True)
    extractor = TransparentObjectExtractor()
    rospy.spin()

if __name__ == '__main__':
    main()

