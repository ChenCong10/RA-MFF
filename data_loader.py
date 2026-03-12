# data loader
from __future__ import print_function, division
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random

#==========================dataset load==========================

class SalObjDataset(Dataset):
	def __init__(self,dataset_dir,transforms_,rgb=True):
		self.dataset_dir=dataset_dir
		self.file_list = os.listdir(self.dataset_dir)
		self.transform = transforms.Compose(transforms_)
		self.rgb=rgb
	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		
		

		if self.rgb:
			sourceA_dir = "/home/ccwydlq10/code/dataset/MFF/Road_MF/High_resolution/Background"
			sourceB_dir = "/home/ccwydlq10/code/dataset/MFF/Road_MF/High_resolution/foreground"
			groundtruth_dir = "/home/ccwydlq10/code/dataset/MFF/Road_MF/GT_high"

			# 选择一张图片
			base_to_file_A = {os.path.splitext(f)[0]: f for f in os.listdir(sourceA_dir)}
			base_to_file_B = {os.path.splitext(f)[0]: f for f in os.listdir(sourceB_dir)}
			base_to_file_GT = {os.path.splitext(f)[0]: f for f in os.listdir(groundtruth_dir)}

			# 获取共同基名列表
			common_bases = list(set(base_to_file_A) & set(base_to_file_B) & set(base_to_file_GT))
			if not common_bases:
				raise ValueError("No common filenames found across all directories.")

			# 随机选择一个共同基名
			selected_base = random.choice(common_bases)

			# 构造对应的文件路径
			img1_path = os.path.join(sourceA_dir, base_to_file_A[selected_base])
			img2_path = os.path.join(sourceB_dir, base_to_file_B[selected_base])
			label_path = os.path.join(groundtruth_dir, base_to_file_GT[selected_base])
		

		img1 = Image.open(img1_path).convert('RGB' if self.rgb else 'L')
		img2 = Image.open(img2_path).convert('RGB' if self.rgb else 'L')
		label = Image.open(label_path).convert('RGB' if self.rgb else 'L')

		# 其他处理
		target_size = 128  # 根据你的模型输出尺寸来设置
		img1 = img1.resize((target_size, target_size), Image.BICUBIC)
		img2 = img2.resize((target_size, target_size), Image.BICUBIC)
		label = label.resize((target_size, target_size), Image.BICUBIC)


		# 水平翻转
		if random.random() < 0.5:
			img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
			img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
			label = label.transpose(Image.FLIP_LEFT_RIGHT)

		img1 = self.transform(img1)
		img2 = self.transform(img2)
		label = self.transform(label)

		return img1, img2, label



class DataTest(Dataset):
    def __init__(self,testData_dir,transforms_):
        self.testData_dir=testData_dir
        self.file_list=os.listdir(testData_dir)
        self.transform = transforms.Compose(transforms_)
    def __getitem__(self, idx):
        image1= Image.open(self.testData_dir+"/"+self.file_list[idx]+"/"+self.file_list[idx]+"-A.jpg")
        image2= Image.open(self.testData_dir+"/"+self.file_list[idx]+"/"+self.file_list[idx]+"-B.jpg")
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1,image2
    def __len__(self):
        return len(self.file_list)

