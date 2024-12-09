import cv2
import numpy as np

def extract_transparent_objects(image_path, lower_hsv, upper_hsv, target_size=(640, 480)):
    image = cv2.imread(image_path)

    # 调整图像到固定大小
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 将 BGR 转换为 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # 使用形态学操作清理掩码
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 提取透明物体
    transparent_objects = cv2.bitwise_and(image, image, mask=mask_cleaned)

    return transparent_objects

# 设置 HSV 阈值 (根据需要调整)
lower_hsv = np.array([0, 00, 120])  # 低阈值 (Hue, Saturation, Value)
upper_hsv = np.array([255, 50, 255])  # 高阈值

# 提取透明物体
result = extract_transparent_objects('/home/xuyuan/pan/data/IMG_8505.JPG', lower_hsv, upper_hsv, target_size=(640, 480))

# 显示结果
cv2.imshow('Transparent Objects', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

