import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('/home/xuyuan/pan/data/IMG_8518.JPG')

# 将图像从BGR转换为HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 分离HSV通道
h_channel, s_channel, v_channel = cv2.split(hsv_image)

# 创建直方图
plt.figure(figsize=(12, 6))

#H通道直方图
plt.subplot(3, 1, 1)
plt.hist(h_channel.ravel(), bins=180, range=[0, 180], color='red', alpha=0.7)
plt.title('Hue Histogram', fontsize=18)  # 设置标题字体大小
plt.xlim([0, 180])
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度标签大小

# S通道直方图
plt.subplot(3, 1, 2)
plt.hist(s_channel.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
plt.title('Saturation Histogram', fontsize=18)  # 设置标题字体大小
plt.xlim([0, 256])
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度标签大小

# V通道直方图
plt.subplot(3, 1, 3)
plt.hist(v_channel.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Value Histogram', fontsize=18)  # 设置标题字体大小
plt.xlim([0, 256])
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度标签大小

plt.tight_layout()
plt.show()

