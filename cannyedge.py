import cv2
import numpy as np

# 读取图像
image_path = '/home/xuyuan/pan/data/D9C9A8DDE1C761A84AA1640F107E0840.png'  # 替换为您的图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, (360,640), interpolation=cv2.INTER_AREA)
# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用 Canny 边缘检测
low_threshold = 100  # 低阈值
high_threshold = 200  # 高阈值
edges = cv2.Canny(gray_image, low_threshold, high_threshold)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓
# 创建一个与原图像相同大小的空白图像
contour_image = np.zeros_like(image)

# 绘制轮廓
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽为2

# 将轮廓图像与原图像合并
combined_image = cv2.addWeighted(image, 0.7, contour_image, 0.3, 0)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Canny Edges", edges)
cv2.imshow("Contours on Original Image", combined_image)
cv2.imshow("gray images", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

