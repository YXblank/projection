import cv2
import numpy as np

# 1. 读取图像
image = cv2.imread('/home/xuyuan/pan/资料/glass.png')
# 2. 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 二值化处理
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 4. 提取轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 创建掩膜
mask = np.zeros_like(gray)

if contours:
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# 填补图像
filled_image = cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Filled Image", filled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
