import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 加载并预处理图片
def load_image(image_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加 batch 维度

# 生成决策图：将两张图片的像素差异作为判断依据
def generate_decision_map(image1, image2, threshold=0.1):
    """
    根据像素差异生成决策图。如果差异超过阈值，则在决策图中设为 1，否则为 0。
    """
    difference = torch.abs(image1 - image2)  # 计算两张图片的绝对差异
    decision_map = (difference > threshold).float()  # 阈值处理，生成二值化的决策图
    decision_map = torch.mean(decision_map, dim=1, keepdim=True)  # 将 RGB 合并为单通道
    return decision_map

# 示例路径（需要替换为实际图片路径）
image_path1 = '/home/ccwydlq10/code/Our/24.9.25/source image/image A/1.png'
image_path2 = '/home/ccwydlq10/code/Our/24.9.25/source image/image B/1.png'

# 加载图片
image1 = load_image(image_path1)
image2 = load_image(image_path2)

# 生成决策图
decision_map = generate_decision_map(image1, image2)

# 保存决策图为图片
output_path = '/home/ccwydlq10/code/Our/24.9.25/source image/result/decision_map1.png'
transforms.ToPILImage()(decision_map.squeeze(0)).save(output_path)

print(f"决策图已保存到 {output_path}")
