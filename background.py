import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/xuyuan/pan/data/IMG_8488.JPG')

# 检查图像是否成功加载
if image is None:
    print("无法加载图像，请检查文件路径。")
else:
    # 打印图像的形状
    print(f'图像形状: {image.shape}')  # 例如 (高, 宽, 通道)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 寻找灰度峰值
    gray_peak_value = np.max(gray_image)

    # 创建掩膜，提取低于灰度峰值的像素点
    mask = gray_image < gray_peak_value

    # 将掩膜应用于原图像，提取背景区域
    background_pixels = image[mask]
    print("图像形状:", background_pixels.shape)

     #打印背景区域的形状
    print("提取的背景区域形状:", background_pixels.shape)

    # 将原图像转换为 HSV 格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 使用掩膜提取背景区域的 HSV 值
    background_hsv = hsv_image[mask]

    # 计算平均 HSV 值
    if background_hsv.size > 0:  # 确保有背景像素
        average_hsv = np.mean(background_hsv, axis=0)
        print("背景区域的平均 HSV 值:", average_hsv)
    else:
        print("没有提取到背景区域的像素。")

