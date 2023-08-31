import os
import glob
import random
import numpy as np
from PIL import Image


# # 加载保存的 mask 数组
# mask = np.load("npy/00000.npy")
#
# # 还原为原始范围
# mask = (mask + 1) * 127.5
# mask = np.clip(mask, 0, 255)
# mask = mask.astype(np.uint8)
#
# # 创建 PIL 图像对象
# mask_image = Image.fromarray(mask, mode='L')
#
# # 保存为 PNG 格式
# mask_image.save("restored_mask.png")


def png_to_npy(mask_path, dest):
    gray_mask = Image.open(mask_path).convert('L')
    mask_arr = np.array(gray_mask, dtype=np.float32)
    mask = mask_arr / 127.5 - 1
    np.save(dest + mask_path.split("\\")[-1].split(".")[0] + ".npy", mask)


def transfer(path, dest, ratio):
    pass


if __name__ == "__main__":
    # all_path = glob.glob(r"E:\Dataset\MICCAI2023 USehance\low_quality_images\*.png")
    # for i in all_path:
    #     png_to_npy(i, "data/predict/real/")

    all_path = glob.glob("E:\\Dataset\\MICCAI2023 USehance\\train_datasets\\*\\*\\*.png")
    for i in all_path:
        if i.split("\\")[-2] == "high_quality":
            if random.random() > 0.87:
                png_to_npy(i, "../data/val/B/")
                png_to_npy(i.replace("high_quality", "low_quality"), "../data/val/A/")
            else:
                png_to_npy(i, "../data/train/B/")
                png_to_npy(i.replace("high_quality", "low_quality"), "../data/train/A/")

    # predictData_path = glob.glob(r"E:\Dataset\MICCAI2023 USehance\low_quality_images\*.png")
    # for i in predictData_path:
    #     png_to_npy(i, "data/predict/real/")

    # all_path = glob.glob("C:\\Users\\zzduo\\Desktop\\FILE\\超声图片的原始数据集\\train_enhance_datasets\\*\\*\\*.png")
    # for i in all_path:
    #     if i.split("\\")[-2] == "high_quality":
    #         png_to_npy(i, "data/train/B/")
    #         png_to_npy(i.replace("high_quality", "low_quality"), "data/train/A/")

"""
    # 加载保存的 mask 数组
    mask = np.load("data/train_kidney_liver/A/0827.npy")

    # 还原为原始范围
    mask = (mask + 1) * 127.5
    mask = np.clip(mask, 0, 255)
    mask = mask.astype(np.uint8)

    # 创建 PIL 图像对象
    mask_image = Image.fromarray(mask, mode='L')

    # 保存为 PNG 格式
    mask_image.save("restored_mask.png")
"""
