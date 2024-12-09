import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
target_size=(640, 480)


    # 调整图像到固定大小
image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# 将图像转换为 HSV 色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义 HSV 颜色范围（根据需要调整）
lower_hue = np.array([0, 00, 120])  # 这里是一个示例值
upper_hue = np.array([255, 50, 255])   # 这里是一个示例值

# 创建掩码以提取指定颜色
mask = cv2.inRange(hsv_image, lower_hue, upper_hue)

# 使用 Canny 边缘检测
edges = cv2.Canny(mask, threshold1=100, threshold2=200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始掩码上绘制轮廓
contour_mask = np.zeros_like(mask)
cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

# 将轮廓掩码应用到原始图像
contour_region = cv2.bitwise_and(image, image, mask=contour_mask)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('HSV Image', hsv_image)
cv2.imshow('Mask', mask)
cv2.imshow('Edges', edges)
cv2.imshow('Contour Mask', contour_mask)
cv2.imshow('Contour Region', contour_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

