import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载彩色图像
image_path = '/home/xuyuan/pan/data/0EA965B00AFD58A9C435C777D80D97F3.png'
image = cv2.imread(image_path)
image = cv2.resize(image,(360,640), interpolation=cv2.INTER_AREA)

# 检查图像是否成功加载
if image is None:
    print("Error: Could not load image.")
else:
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 找到最高峰值及其对应的强度值
    max_value_index = np.argmax(hist)  # 最高峰的强度值索引
    max_pixel_count = hist[max_value_index]  # 最高峰的像素计数

    print(f"Highest peak value: {max_pixel_count} at intensity level: {max_value_index}")

    # 找到在图像中对应该强度值的所有像素的坐标
    coordinates = np.argwhere(gray_image == max_value_index)

    # 可视化：在原图上标记出这些坐标
    marked_image = image.copy()
    for coord in coordinates:
        cv2.circle(marked_image, (coord[1], coord[0]), 2, (0, 0, 255), -1)  # 用红色标记

    # 显示标记后的图像
    cv2.imshow('Marked Image', marked_image)

    # 绘制直方图
    plt.plot(hist)
    plt.title('Intensity Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')

    plt.show()  # 显示直方图

    cv2.waitKey(0)  # 等待按键输入
    cv2.destroyAllWindows()  # 关闭所有窗口

