import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# 读取图像并转换为灰度图
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(gray, (360,640), interpolation=cv2.INTER_AREA)

# 将图像量化到 8 个灰度级别
image = (image // 32).astype(np.uint8)

# 计算灰度共生矩阵
glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

# 提取纹理特征
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print("对比度: ", contrast)
print("同质性: ", homogeneity)
print("能量: ", energy)
print("相关性: ", correlation)

