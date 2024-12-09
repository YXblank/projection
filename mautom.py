import cv2
import numpy as np

def region_growing(image, seeds, threshold):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    pixel_list = list(seeds)  # 使用种子点初始化列表
    for seed in seeds:
        mask[seed[1], seed[0]] = 1  # 标记种子点

    while pixel_list:
        x, y = pixel_list.pop(0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not mask[ny, nx]:
                    if np.linalg.norm(image[ny, nx] - image[y, x]) < threshold:
                        mask[ny, nx] = 1
                        pixel_list.append((nx, ny))

    return mask

def get_seed(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed_points
        seed_points.append((x, y))
        print(f'Seed point selected: {(x, y)}')

def extract_transparent_objects(image_path, lower_hsv, upper_hsv, target_size=(360, 640)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    transparent_objects = cv2.bitwise_and(image, image, mask=mask_cleaned)
    return transparent_objects, mask_cleaned

# 设置 HSV 阈值
lower_hsv = np.array([00, 0, 50])
upper_hsv = np.array([250, 250, 255])

# 提取透明物体和掩码
transparent_objects, cleaned_mask = extract_transparent_objects('/home/xuyuan/pan/data/test3.png', lower_hsv, upper_hsv)

# 初始化全局变量
seed_points = []
threshold = 100

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_seed)

# 显示图像并等待用户点击
while True:
    cv2.imshow('Image', transparent_objects)

    if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
        break

# 用户完成点击后，执行区域生长
if seed_points:
    segmented_mask = region_growing(transparent_objects, seed_points, threshold)
    segmented_image = cv2.bitwise_and(transparent_objects, transparent_objects, mask=segmented_mask)

    # 显示结果
    cv2.imshow('Segmented Image', segmented_image * 255)
    cv2.imshow('Segmented Region', segmented_mask * 255)  # 将掩码显示为白色区域

    cv2.waitKey(0)  # 等待按键以查看结果

cv2.destroyAllWindows()

