import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter

def ConvertRGBtoYUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def ConvertYUVtoRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

def generate_shape_mask(shape_type, row, column):
    binary_image = np.zeros((row, column), dtype=np.uint8)

    if shape_type == 1:  # 六边形
        center = (column // 2, row // 2)
        radius = min(row, column) // 4
        angle = np.linspace(0, 2 * np.pi, 7)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        binary_image = cv2.fillPoly(binary_image, [np.vstack((x, y)).T.astype(np.int32)], 1)

    elif shape_type == 2:  # 椭圆
        major_axis = min(row, column) // 2
        minor_axis = min(row, column) // 3
        center_x = column // 2
        center_y = row // 2
        y_grid, x_grid = np.ogrid[:row, :column]
        binary_image = ((x_grid - center_x)**2 / major_axis**2 + (y_grid - center_y)**2 / minor_axis**2) <= 1

    elif shape_type == 3:  # 圆形
        radius = min(row, column) // 4
        center_x = column // 2
        center_y = row // 2
        y_grid, x_grid = np.ogrid[:row, :column]
        binary_image = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2

    return binary_image

def apply_shape_mask(img1, img2):
    row, column = img1.shape
    shape_type = random.randint(1, 3)
    binary_image = generate_shape_mask(shape_type, row, column)

    sigma = 5
    M = gaussian_filter(img1, sigma)
    M1 = binary_image * M
    M11 = binary_image * img1
    M2 = (1 - binary_image) * M
    M22 = (1 - binary_image) * img1

    I1 = M1 + M22
    I2 = M2 + M11

    return I1, I2

def process_image(img1, img2):
    img1 = img1.astype(np.float64) / 255.0

    if img1.shape[2] > 1:
        A_YUV = ConvertRGBtoYUV(img1)
        f1 = A_YUV[:, :, 0]
    else:
        f1 = img1

    if img2.shape[2] > 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = img2.astype(np.float64) / 255.0
    else:
        img2 = img2.astype(np.float64) / 255.0

    I1, I2 = apply_shape_mask(f1, img2)

    if img1.shape[2] > 1:
        F1_YUV = np.zeros((f1.shape[0], f1.shape[1], 3))
        F1_YUV[:, :, 0] = I1
        F1_YUV[:, :, 1:] = A_YUV[:, :, 1:]
        F1 = ConvertYUVtoRGB(F1_YUV)

        F2_YUV = np.zeros((f1.shape[0], f1.shape[1], 3))
        F2_YUV[:, :, 0] = I2
        F2_YUV[:, :, 1:] = A_YUV[:, :, 1:]
        F2 = ConvertYUVtoRGB(F2_YUV)
    else:
        F1, F2 = I1, I2

    return (F1 * 255).astype(np.uint8), (F2 * 255).astype(np.uint8)

class ImageMaskModule:
    def __init__(self):
        pass

    def apply_to_images(self, img1, img2):
        return process_image(img1, img2)

if __name__ == "__main__":
    # 示例代码
    img1 = cv2.imread("path/to/image1.png")
    img2 = cv2.imread("path/to/image2.png")
    module = ImageMaskModule()
    F1, F2 = module.apply_to_images(img1, img2)