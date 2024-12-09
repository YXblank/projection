import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(gray, (360,640), interpolation=cv2.INTER_AREA)

# 计算LBP特征
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

# 显示LBP图像
plt.imshow(lbp_image, cmap='gray')
plt.title('LBP Image')
plt.show()

# 计算LBP直方图
lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
lbp_hist = lbp_hist.astype('float')
lbp_hist /= lbp_hist.sum()  # 归一化直方图

print("LBP直方图: ", lbp_hist)

