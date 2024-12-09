import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('/home/xuyuan/pan/data/IMG_8505.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(gray, (360,640), interpolation=cv2.INTER_AREA)


# 1. 计算 LBP（局部二值模式）特征
radius = 1  # 邻域的半径
n_points = 8 * radius  # 邻域点的数目
lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

# 2. 计算 GLCM（灰度共生矩阵）特征
# 首先量化图像，将灰度级别缩小为8级
image_quantized = (image // 32).astype(np.uint8)  # 8级灰度

# 计算GLCM矩阵，使用距离为1像素，考虑四个方向（0°, 45°, 90°, 135°）
glcm = graycomatrix(image_quantized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

# 3. 提取 GLCM 特征
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')  # 使用同质性
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

# 4. 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示原图
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# 显示 LBP 图像
axes[1].imshow(lbp_image, cmap='gray')
axes[1].set_title("LBP Image")
axes[1].axis('off')

# 显示同质性（Homogeneity）矩阵
for i in range(4):  # 4 个方向
    ax = axes[2]
    ax.imshow(glcm[:, :, 0, i], cmap='hot')  # 显示每个方向的GLCM
    ax.set_title(f"GLCM - Homogeneity ({i * 45}°)")
    ax.axis('off')

plt.show()

# 输出 GLCM 的统计特征
print("GLCM 特征 - 对比度: ", contrast)
print("GLCM 特征 - 同质性: ", homogeneity)
print("GLCM 特征 - 能量: ", energy)
print("GLCM 特征 - 相关性: ", correlation)

