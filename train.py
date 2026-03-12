import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import trange
from Net1.loss import DiceLoss, SSIM
from Net1.model import FusionModel
from Net1.cc_dataset import CustomSceneDataset  

# 保存输出的预测决策图
def save_output_images(output, save_dir='train_images', batch_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    output = output.cpu().detach()
    save_path = os.path.join(save_dir, f"output_{batch_idx}.png")
    save_image(output, save_path, normalize=True)

# 调整后的混合损失函数
def MixLoss(GT, NetOut):
    loss_ssim = SSIM()
    loss_L1 = nn.L1Loss()
    loss_dice = DiceLoss()
    # 调整权重：L1:0.5, Dice:0.2, SSIM:0.3
    loss = 0.5 * loss_L1(NetOut, GT) + 0.2 * loss_dice(NetOut, GT) + 0.3 * (1 - loss_ssim(GT, NetOut).item())
    return loss

# 单个 epoch 的训练逻辑（决策图训练版）
def train_one_epoch(model, loader, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    for batch_idx, (image1, image2, groundtruth) in enumerate(loader):
        image1, image2, groundtruth = image1.to(device), image2.to(device), groundtruth.to(device)
        groundtruth = F.interpolate(groundtruth, size=(112, 112), mode='bilinear', align_corners=False)

        optimizer.zero_grad()

        with autocast():
            output = model(image1, image2)
            # 如果模型输出未经过激活函数，可考虑注释或取消下面这行：
            # output = torch.relu(output)
            loss = MixLoss(groundtruth, output)

        loss /= accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        if batch_idx % 10 == 0:
            save_output_images(output, batch_idx=batch_idx)

        running_loss += loss.item()

    return running_loss / len(loader)

# 单个 epoch 的验证逻辑（决策图验证版）
def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for image1, image2, groundtruth in loader:
            image1, image2, groundtruth = image1.to(device), image2.to(device), groundtruth.to(device)
            groundtruth = F.interpolate(groundtruth, size=(112, 112), mode='bilinear', align_corners=False)

            output = model(image1, image2)
            loss = MixLoss(groundtruth, output)
            running_loss += loss.item()

    return running_loss / len(loader)

# 主函数
def main():
    batch_size = 8
    num_epochs = 100
    # 使用较低的学习率以稳定训练
    learning_rate = 5e-5
    accumulation_steps = 8

    best_val_loss = float('inf')
    device = torch.device('cuda:0')

    # 数据预处理
    transforms_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    project_dir = os.getcwd()
    train_dir = os.path.join(project_dir, '/home/ccwydlq10/code/dataset/MFF/VOCdevkit/train')
    val_dir = os.path.join(project_dir, '/home/ccwydlq10/code/dataset/MFF/VOCdevkit/test')

    train_dataset = CustomSceneDataset(train_dir, transform=transforms_, need_crop=False, need_augment=False)
    val_dataset = CustomSceneDataset(val_dir, transform=transforms_, need_crop=False, need_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 使用决策图模型（FusionModel 输出决策图）
    model = FusionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = GradScaler()

    with trange(num_epochs, desc="Training Progress", unit="epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, accumulation_steps)
            val_loss = validate_one_epoch(model, val_loader, device)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/best_model.pth')

            if epoch == num_epochs - 1:
                torch.save(model.state_dict(), 'models/last_model.pth')

            torch.cuda.empty_cache()
            tepochs.set_postfix(train_loss=train_loss, val_loss=val_loss)

if __name__ == "__main__":
    main()
