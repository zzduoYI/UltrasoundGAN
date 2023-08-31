import cv2
import glob
import numpy as np



def rever(image):
    # 定义三角形顶点坐标
    triangle1_points = np.array([[0, 0], [90, 0], [0, 150]])
    triangle2_points = np.array([[0, 0], [0, 150], [90, 0]])
    # 创建一个与原始图像大小相同的黑色图像作为输出
    output_image = np.zeros_like(image)
    # 循环遍历图像的每个像素
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # 如果像素位于三角形1内部，则保持不变
            if cv2.pointPolygonTest(triangle1_points, (x, y), False) >= 0:
                output_image[y, x] = image[y, x]
            else:
                # 计算镜像翻转后的像素位置
                mirrored_x = image.shape[1] - 1 - x
                mirrored_y = y
                # 如果像素位于三角形2内部，则设置为黑色
                if cv2.pointPolygonTest(triangle2_points, (mirrored_x, mirrored_y), False) >= 0:
                    output_image[y, x] = [0, 0, 0]
                else:
                    output_image[y, x] = image[mirrored_y, mirrored_x]
    return output_image


if __name__ == "__main__":
    """
    # 单张图片测试
    image = cv2.imread('0825.png')
    output = rever(image)
    cv2.imwrite("rever.png", output)
    """
    # kidney 肾镜像翻转代码
    # liver 肺镜像翻转代码
    files = glob.glob(r"E:\Dataset\MICCAI2023 USehance\train_datasets\liver\high_quality\*")
    for i in files:
        image_high_quality = cv2.imread(i)
        rever_image_high_quality = rever(image_high_quality)
        cv2.imwrite(i.split(".")[0] + "_reversal_.png", rever_image_high_quality)
        image_low_quality = cv2.imread(i.replace("high_quality", "low_quality"))
        rever_image_low_quality = rever(image_low_quality)
        cv2.imwrite(i.replace("high_quality", "low_quality").split(".")[0] + "_reversal_.png", rever_image_low_quality)
