import cv2
import numpy as np

# 读取图像并转换为灰度图像
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (360,640), interpolation=cv2.INTER_AREA)

# 找到轮廓
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建掩膜
mask = np.zeros_like(gray)

# 在掩膜上绘制轮廓
if contours:
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# 使用Canny边缘检测
edges = cv2.Canny(gray, 100, 200)

# 膨胀边缘以连接缺口
kernel = np.ones((20, 20), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# 在膨胀的边缘上绘制轮廓
mask[dilated_edges > 0] = 255

# 使用 inpainting 填补图像
filled_image = cv2.inpaint(gray, mask, inpaintRadius=50, flags=cv2.INPAINT_NS)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Dilated Edges", dilated_edges)
cv2.imshow("Mask", mask)
cv2.imshow("Filled Image", filled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

