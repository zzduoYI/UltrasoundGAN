import cv2
import glob
import random
import numpy as np


def distort_center(image, width, height, strength, center_x=128, center_y=128):
    # 获取图像尺寸
    image_height, image_width = image.shape[:2]

    # 定义一个矩形区域，表示中心部分
    top_left_x = max(0, int(center_x - width / 2))
    top_left_y = max(0, int(center_y - height / 2))

    bottom_right_x = min(image_width, int(center_x + width / 2))
    bottom_right_y = min(image_height, int(center_y + height / 2))

    # 获取中心部分的图像
    center_patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # 构造扭曲的网格坐标
    num_rows, num_cols = center_patch.shape[:2]
    map_x, map_y = np.meshgrid(np.linspace(0, num_cols - 1, num_cols),
                               np.linspace(0, num_rows - 1, num_rows))

    # 添加形变效果
    map_x = map_x + strength * np.sin(map_y / height * np.pi)
    map_y = map_y + strength * np.cos(map_x / width * np.pi)

    # 扭曲中心部分图像
    distorted_patch = cv2.remap(center_patch, map_x.astype(np.float32), map_y.astype(np.float32),
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 将扭曲后的中心部分放回原图像中
    distorted_image = image.copy()
    distorted_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = distorted_patch

    return distorted_image


if __name__ == "__main__":
    """
    单独图片测试代码
    
    # 读取图像
    image_path = '1213.png'
    image = cv2.imread(image_path)
    # 定义中心部分的参数
    width = 256  # 中心部分的宽度
    height = 256  # 中心部分的高度
    strength = 20  # 扭曲的强度
    # 进行扭曲形变
    distorted_image = distort_center(image, width, height, strength)
    # 显示原图和扭曲后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Distorted Image', distorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    """
    # breast 胸部扭曲变形的代码
    # bottom_right_x = 233
    files = glob.glob(r"E:\Dataset\MICCAI2023 USehance\train_datasets\breast\high_quality\*")
    for i in files:
        image_high_quality = cv2.imread(i)
        random_number = random.randint(20, 50)  # 生成随机整数
        distorted_image_high_quality = distort_center(image_high_quality, 256, 256, random_number)
        cv2.imwrite(i.split(".")[0] + "_twist_.png", distorted_image_high_quality)
        image_low_quality = cv2.imread(i.replace("high_quality", "low_quality"))
        distorted_image_low_quality = distort_center(image_low_quality, 256, 256, random_number)
        cv2.imwrite(i.replace("high_quality", "low_quality").split(".")[0] + "_twist_.png", distorted_image_low_quality)
    """

    """
    # carotid 颈动脉扭曲变形的代码
    files = glob.glob(r"E:\Dataset\MICCAI2023 USehance\train_datasets\carotid\high_quality\*")
    for i in files:
        image_high_quality = cv2.imread(i)
        random_number = random.randint(20, 50)  # 生成随机整数
        distorted_image_high_quality = distort_center(image_high_quality, 256, 256, random_number)
        cv2.imwrite(i.split(".")[0] + "_twist_.png", distorted_image_high_quality)
        image_low_quality = cv2.imread(i.replace("high_quality", "low_quality"))
        distorted_image_low_quality = distort_center(image_low_quality, 256, 256, random_number)
        cv2.imwrite(i.replace("high_quality", "low_quality").split(".")[0] + "_twist_.png", distorted_image_low_quality)
    """

    """
    # thyroid 甲状腺扭曲变形代码
    files = glob.glob(r"E:\Dataset\MICCAI2023 USehance\train_datasets\thyroid\high_quality\*")
    for i in files:
        image_high_quality = cv2.imread(i)
        random_number = random.randint(20, 50)  # 生成随机整数
        distorted_image_high_quality = distort_center(image_high_quality, 256, 256, random_number)
        cv2.imwrite(i.split(".")[0] + "_twist_.png", distorted_image_high_quality)
        image_low_quality = cv2.imread(i.replace("high_quality", "low_quality"))
        distorted_image_low_quality = distort_center(image_low_quality, 256, 256, random_number)
        cv2.imwrite(i.replace("high_quality", "low_quality").split(".")[0] + "_twist_.png", distorted_image_low_quality)
    """



