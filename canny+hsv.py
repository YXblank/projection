import cv2
import numpy as np
def region_growing(image, seed, threshold):
    # 获取图像的大小
    h, w = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建一个与输入图像相同大小的标记图
    segmented = np.zeros((h, w), dtype=np.uint8)
    
    # 定义一个队列来存放待处理的像素
    pixel_list = [seed]
    
    # 获取种子点的强度值
    seed_value = gray_image[seed[1], seed[0]]
    
    while pixel_list:
        x, y = pixel_list.pop(0)
        
        # 检查边界条件
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        
        # 检查当前像素是否已经被标记
        if segmented[y, x] == 1:
            continue
        
        # 检查强度值条件
        if abs(int(gray_image[y, x]) - int(seed_value)) < threshold:  # 使用强度值阈值
            segmented[y, x] = 1  # 标记为已处理
            
            # 添加相邻像素到列表
            pixel_list.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    
    return segmented


def get_seed(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed_point
        seed_point = (x, y)
        print(f'Seed point selected: {seed_point}')
# 选择种子点和强度阈值
#seed_point = (100, 100)  # 根据需要修改坐标
#intensity_threshold = 15  # 设置强度值阈值



def extract_transparent_objects(image_path, lower_hsv, upper_hsv):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (360,640), interpolation=cv2.INTER_AREA)
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

def find_and_draw_contours(image, mask):
    # 使用 Canny 边缘检测
    edges = cv2.Canny(mask, threshold1=100, threshold2=200)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制轮廓
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), thickness=2)

    return contour_image, contours

# 设置 HSV 阈值 (根据需要调整)
lower_hsv = np.array([0, 0, 150])  # 低阈值 (Hue, Saturation, Value)# hue:0-20
upper_hsv = np.array([180, 50, 255])  # 高阈值

# 提取透明物体和掩码
transparent_objects, cleaned_mask = extract_transparent_objects('/home/xuyuan/pan/data/mbottle.png', lower_hsv, upper_hsv)

# 查找并绘制轮廓
contour_image, contours = find_and_draw_contours(transparent_objects, cleaned_mask)
# 进行区域增长
#segmented_image = region_growing(transparent_objects, seed_point, intensity_threshold)
# 创建窗口并设置回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_seed)

# 显示图像并等待用户点击
while True:
    cv2.imshow('Image', transparent_objects)
    if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
        break

# 应用区域增长法
threshold = 15
segmented_mask = region_growing(transparent_objects, seed_point, threshold)

# 显示分割结果
cv2.imshow('Segmented Region', segmented_mask * 255)
# 显示结果
#cv2.imshow('Segmented Image', segmented_image * 255)  # 转换为可视化形式

# 显示结果
cv2.imshow('Transparent Objects', transparent_objects)
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

