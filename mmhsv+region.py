import cv2
import numpy as np
import random
import time
from multiprocessing import Pool
start_time = time.time()
def generate_random_seed_points(image, num_seeds):
    height, width = image.shape[:2]
    seeds = []
    
    for _ in range(num_seeds):
        x = random.randint(width // 4, 3 * width // 4)
        y = random.randint(height // 4, 3 * height // 4)
        seeds.append((x, y))
        
    return seeds

def region_growing(image, seed, threshold):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x, y = seed
    seed_value = image[y, x]
    
    pixel_list = [seed]
    mask[y, x] = 1

    while pixel_list:
        x, y = pixel_list.pop(0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not mask[ny, nx]:
                    if np.linalg.norm(image[ny, nx] - seed_value) < threshold:
                        mask[ny, nx] = 1
                        pixel_list.append((nx, ny))

    return mask

def process_seed(seed):
    return region_growing(transparent_objects, seed, threshold=40)

def extract_transparent_objects(image_path, lower_hsv, upper_hsv, target_size=(640, 480)):
    image = cv2.imread(image_path)

    # 调整图像到固定大小
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    transparent_objects = cv2.bitwise_and(image, image, mask=mask_cleaned)
    return transparent_objects, mask_cleaned


# 设置 HSV 阈值

lower_hsv = np.array([50, 0, 50])#0, 0, 150   #20, 0, 50for opaque glass surrounding
upper_hsv = np.array([150, 50, 255])#180, 50, 255])   #150, 50, 155]for green glass surrounding
image = cv2.imread('/home/xuyuan/pan/data/plate.png')
image = cv2.resize(image, (360,640), interpolation=cv2.INTER_AREA)

num_seeds = 20
transparent_objects, cleaned_mask = extract_transparent_objects('/home/xuyuan/pan/data/plate.png', lower_hsv, upper_hsv, target_size=(360,640))

threshold = 15
seed_points = generate_random_seed_points(transparent_objects, num_seeds)

# 创建一个进程池并并行处理区域生长
if __name__ == '__main__':
    with Pool(processes=8) as pool:  # 可根据系统调整进程数量
        segmented_masks = pool.map(process_seed, seed_points)

    # 合并所有分割的掩码
    final_mask = np.zeros((transparent_objects.shape[0], transparent_objects.shape[1]), dtype=np.uint8)
    for mask in segmented_masks:
        final_mask = cv2.bitwise_or(final_mask, mask)

    segmented_image = cv2.bitwise_and(transparent_objects, transparent_objects, mask=final_mask)

result_image = np.clip(transparent_objects.astype(np.float32) - segmented_image.astype(np.float32), 0, 255).astype(np.uint8)

minus_image = np.clip(image.astype(np.float32) - transparent_objects.astype(np.float32), 0, 255).astype(np.uint8) 
# 显示结果
cv2.imshow('Result Image', result_image)
# 可视化分割结果
cv2.imshow(f'Segmented Region from Seed ', segmented_image)
# 显示分割结果
#cv2.imshow('Segmented Region', segmented_mask * 255)
# 显示结果
#cv2.imshow('Segmented Image', segmented_image * 255)  # 转换为可视化形式
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")
# 显示结果
cv2.imshow('Transparent Objects', transparent_objects)
cv2.imshow('minus_image Image', minus_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

