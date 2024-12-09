import cv2
import matplotlib.pyplot as plt
import numpy as np
# 加载彩色图像
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
image = cv2.resize(image, (360,640), interpolation=cv2.INTER_AREA)

# 转换为灰度图像以获取强度信息
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示灰度图像
cv2.imshow('Gray Image', gray_image)

# 输出强度值
print(gray_image)

# 计算灰度直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# 创建强度值数组
intensity_values = np.arange(256)

plt.plot(hist, color='red')
plt.title('Intensity Histogram of Bowl',fontsize=20)
plt.xlabel('Intensity Value',fontsize=18)
plt.ylabel('Pixel Count',fontsize=18)
plt.tick_params(axis='both',labelsize=16)
plt.xlim([0, 255])  # 设置X轴范围
plt.show()

# 释放窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


