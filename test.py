# test2.py
import os
import sys
import glob
import time
import cv2
import torch
import multiprocessing
from tqdm import tqdm
from torch import einsum
from Net1.model import FusionModel as Network
from loosses import Consistency
import Net1.cc_dataset as DLr
from torch.utils.data import DataLoader
from Net1.loss import GPUorCPU

# 多进程配置必须放在最前
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def worker_init(worker_id):
    """子进程内存配置"""
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class Fusion:
    def __init__(self,
                 modelpath='/home/featurize/work/our_work/models/2025-04-22 13.56.21/best_network.pth',
                 dataroot='/home/featurize/work/our_work/test_images',
                 dataset_name='Lytro',
                 threshold=0.001,
                 window_size=5):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME is not None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        result_dir = './results' + savepath
        os.makedirs(result_dir, exist_ok=True)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(
            dataset=eval_data,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,  # 必须启用
            persistent_workers=True,
            worker_init_fn=worker_init
        )

        with torch.no_grad():
            for idx, (A, B) in enumerate(tqdm(eval_loader, desc='Processing')):
                # 确保数据在CPU加载后移动到GPU
                A, B = A.to(self.DEVICE), B.to(self.DEVICE)

                # 调整输入尺寸到16的倍数
                _, _, h, w = A.shape
                new_h = (h // 16) * 16
                new_w = (w // 16) * 16
                if h != new_h or w != new_w:
                    A = F.interpolate(A, (new_h, new_w), mode='bilinear')
                    B = F.interpolate(B, (new_h, new_w), mode='bilinear')

                # 模型推理
                D = model(A, B)
                D = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)
                D = einsum('c w h -> w h c', D[0]).cpu().numpy()

                # 保存结果
                A_img = cv2.imread(eval_list_A[idx])
                B_img = cv2.imread(eval_list_B[idx])
                D_resized = cv2.resize(D, (A_img.shape[1], A_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                IniF = (A_img * D_resized + B_img * (1 - D_resized)).astype('uint8')
                
                base_name = os.path.splitext(os.path.basename(eval_list_A[idx]))[0]
                cv2.imwrite(f"{result_dir}/{base_name}_mask.png", (D_resized * 255).astype('uint8'))
                cv2.imwrite(f"{result_dir}/{base_name}.png", IniF)

if __name__ == '__main__':
    f = Fusion()
    f()