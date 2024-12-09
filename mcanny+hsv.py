import cv2
import numpy as np
import random
import time
start_time = time.time()
def generate_random_seed_points(image, num_seeds):
    height, width = image.shape[:2]  # 获取图像的高度和宽度
    seeds = []
    
    for _ in range(num_seeds):
        x = random.randint(width // 4+20, 3 * width // 4+20)  # 在图像的中心区域随机选择X坐标
        y = random.randint(height // 4+10, 3 * height // 4+10)  # 在图像的中心区域随机选择Y坐标
        seeds.append((x, y))  # 将生成的种子点添加到列表中
        
    return seeds  # 返回生成的种子点列表
def region_growing(image, seed, threshold):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 初始化种子点
    x, y = seed
    seed_value = image[y, x]
    
    pixel_list = [(x, y)]
    mask[y, x] = 1

    while pixel_list:
        # 从列表中弹出一个点
        x, y = pixel_list.pop(0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not mask[ny, nx]:
                    # 计算颜色差异，使用欧几里得距离
                    if np.linalg.norm(image[ny, nx] - seed_value) < threshold:
                        mask[ny, nx] = 1
                        pixel_list.append((nx, ny))  # 确保使用元组形式

    return mask

def get_seed(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed_point
        seed_point = (x, y)
        print(f'Seed point selected: {seed_point}')
# 选择种子点和强度阈值
#seed_point = (100, 100)  # 根据需要修改坐标
#intensity_threshold = 15  # 设置强度值阈值



def extract_transparent_objects(image_path, lower_hsv, upper_hsv, target_size=(640,480)):
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
lower_hsv = np.array([00, 00, 50])  # 低阈值 (Hue, Saturation, Value)# hue:0-20
upper_hsv = np.array([200, 50, 200])  # 高阈值

num_seeds = 4
# 提取透明物体和掩码
transparent_objects, cleaned_mask = extract_transparent_objects('/home/xuyuan/pan/data/mbottle.png', lower_hsv, upper_hsv, target_size=(360,640))

# 查找并绘制轮廓
#contour_image, contours = find_and_draw_contours(transparent_objects, cleaned_mask)
# 进行区域增长
#segmented_image = region_growing(transparent_objects, seed_point, intensity_threshold)
# 创建窗口并设置回调
#cv2.namedWindow('Image')
#cv2.setMouseCallback('Image', get_seed)

# 显示图像并等待用户点击
#while True:
    #cv2.imshow('Image', transparent_objects)
    #if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
        #break

# 应用区域增长法
threshold = 10
seed_points = generate_random_seed_points(transparent_objects, num_seeds)
#segmented_mask = region_growing(transparent_objects, seed_points, threshold)
# 创建一个合并掩码
final_mask = np.zeros((transparent_objects.shape[0], transparent_objects.shape[1]), dtype=np.uint8)
for seed in seed_points:
    segmented_mask = region_growing(transparent_objects, seed, threshold=40)  # 可以根据需要调整阈值
    final_mask = cv2.bitwise_or(final_mask, segmented_mask)  # 合并结果
    # 将掩码应用于原图像以进行可视化
segmented_image = cv2.bitwise_and(transparent_objects, transparent_objects, mask=final_mask)
# 将透明对象和分割图像转换为浮点数
# 执行减法操作
result_image = np.clip(transparent_objects.astype(np.float32) - segmented_image.astype(np.float32), 0, 255).astype(np.uint8)


# 显示结果
cv2.imshow('Result Image', result_image)
# 可视化分割结果
cv2.imshow(f'Segmented Region from Seed {seed}', segmented_image)
# 显示分割结果
cv2.imshow('Segmented Region', final_mask * 255)
# 显示结果
#cv2.imshow('Segmented Image', segmented_image * 255)  # 转换为可视化形式
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")
# 显示结果
cv2.imshow('Transparent Objects', transparent_objects)

#cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

