import cv2
import numpy as np

def extract_transparent_objects(image_path, lower_hsv, upper_hsv, target_size=(640, 480)):
    image = cv2.imread(image_path)

    # 调整图像到固定大小
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 将 BGR 转换为 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # 使用形态学操作清理掩码
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 提取透明物体
    transparent_objects = cv2.bitwise_and(image, image, mask=mask_cleaned)

    return transparent_objects, mask_cleaned

def calculate_histogram(image):
    # 计算灰度直方图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist

def find_peaks(hist):
    # 找到直方图的峰值
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)
    return peaks

def find_and_draw_contours(image, edges):
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制轮廓
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), thickness=2)

    return contour_image, contours

# 设置 HSV 阈值 (根据需要调整)
lower_hsv = np.array([10, 00, 120])  # 低阈值 (Hue, Saturation, Value)
upper_hsv = np.array([105, 30, 255])  # 高阈值

# 提取透明物体和掩码
transparent_objects, cleaned_mask = extract_transparent_objects('/home/xuyuan/pan/data/IMG_8518.JPG', lower_hsv, upper_hsv, )

# 第一次 Canny 边缘检测
first_edges = cv2.Canny(cleaned_mask, 100, 200)

# 计算灰度直方图
hist = calculate_histogram(transparent_objects)

# 找到直方图的峰值
peaks = find_peaks(hist)

# 打印峰值
print("Detected peaks:", peaks)

peak_heights = hist[peaks]

# 获取最大峰值
if len(peaks) > 0:
    max_peak_index = peaks[np.argmax(peak_heights)]
    high_threshold = int(hist[max_peak_index])
else:
    high_threshold = 253  # 默认值，假如没有找到峰值

# 设置低阈值
low_threshold = 220  # 默认值

# 第二次 Canny 边缘检测
second_edges = cv2.Canny(first_edges, low_threshold, high_threshold)

# 查找并绘制轮廓
contour_image, contours = find_and_draw_contours(transparent_objects, second_edges)

# 显示结果
cv2.imshow('Transparent Objects', transparent_objects)
cv2.imshow('Edges from Second Canny', second_edges)
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

