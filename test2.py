import os
import sys
import glob
import time
import cv2

import numpy as np
import torch
# —— 指定用卡、避免碎片化、清空缓存 —— #
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()
from tqdm import tqdm
from torch import einsum
from Net1.model import FusionModel as Network
from loosses import Consistency
import Net1.cc_dataset as DLr
from torch.utils.data import DataLoader
from Net1.loss import GPUorCPU


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)



class Fusion:
    def __init__(self,
                 modelpath='/home/featurize/code/models/2025-05-24 09.49.52/best_network.pth',
                 dataroot='/home/featurize/code/our_work/test_images',
                 dataset_name='Lytro',
                 threshold=0.001,
                 window_size=5,
                 ):
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
            pass

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()

        def count_prelu(m, x, y):
            x = x[0]
            m.total_ops = torch.tensor(x.numel())
            m.total_params = torch.tensor([sum(p.numel() for p in m.parameters())])

        from thop import profile, clever_format
        custom_ops = {torch.nn.PReLU: count_prelu}
        flops, params = profile(
            model,
            inputs=(
                torch.rand(1, 3, 520, 520).to(self.DEVICE),
                torch.rand(1, 3, 520, 520).to(self.DEVICE)
            ),
            custom_ops=custom_ops
        )
        flops, params = clever_format([flops, params], "%.5f")
        print('flops: {}, params: {}\n'.format(flops, params))
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

    def tile_inference(self, model, A, B, patch_size=256, overlap=32):
        _, _, H, W = A.shape
        stride = patch_size - overlap
        out_mask = torch.zeros_like(A[:, :1, :, :])
        count_map = torch.zeros_like(out_mask)

        for i in range(0, H, stride):
            for j in range(0, W, stride):
                h_end = min(i + patch_size, H)
                w_end = min(j + patch_size, W)
                h_start = max(h_end - patch_size, 0)
                w_start = max(w_end - patch_size, 0)

                A_patch = A[:, :, h_start:h_end, w_start:w_end]
                B_patch = B[:, :, h_start:h_end, w_start:w_end]

                with torch.cuda.amp.autocast():  # 混合精度推理
                    D_patch = model(A_patch, B_patch)
                    if D_patch.shape[1] == 3:
                        D_patch = D_patch[:, 0:1, :, :]
                    D_patch = torch.where(D_patch > 0.5, 1., 0.)

                out_mask[:, :, h_start:h_end, w_start:w_end] += D_patch
                count_map[:, :, h_start:h_end, w_start:w_end] += 1

        out_mask = out_mask / count_map.clamp(min=1.0)
        return out_mask
    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        result_dir = './results' + savepath
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)

        cnt = 1
        running_time = []

        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                A = A.to(self.DEVICE)
                B = B.to(self.DEVICE)

                # resize 输入到最近的16倍数（保持输入对齐）
                _, _, h, w = A.shape
                new_h = (h // 16) * 16
                new_w = (w // 16) * 16
                if h != new_h or w != new_w:
                    A = torch.nn.functional.interpolate(A, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    B = torch.nn.functional.interpolate(B, size=(new_h, new_w), mode='bilinear', align_corners=False)

                start_time = time.time()

                # ✅ 使用 patch-wise + autocast 混合精度推理
                D = self.tile_inference(model, A, B, patch_size=256, overlap=32)
                D = self.ConsisVerif(D, threshold)

                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()

                # 读取原图
                A_img = cv2.imread(eval_list_A[cnt - 1])
                B_img = cv2.imread(eval_list_B[cnt - 1])

                # Resize mask 到原图尺寸
                D_resized = cv2.resize(D, (A_img.shape[1], A_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                IniF = A_img * D_resized[..., np.newaxis] + B_img * (1 - D_resized[..., np.newaxis])


                filename = os.path.basename(eval_list_A[cnt - 1])
                name, ext = os.path.splitext(filename)

                mask_path = os.path.join(result_dir, name + '_mask' + ext)
                fusion_path = os.path.join(result_dir, filename)

                # cv2.imwrite(mask_path, D_resized * 255)
                cv2.imwrite(fusion_path, IniF.astype('uint8'))

                cnt += 1
                running_time.append(time.time() - start_time)

        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + result_dir)


if __name__ == '__main__':
    f = Fusion()
    f()
