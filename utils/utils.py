import argparse  # 导入argparse库，用于命令行参数解析
import os  # 导入os库，用于操作系统相关的功能
import torchvision.models as models  # 导入torchvision中的模型库
from Net1.model1 import *  # 从自定义的模型模块(Net1.model1)中导入所有内容

def parse_args():
    """获取命令行参数。"""
    parser = argparse.ArgumentParser(prog='GEU-Net params',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="输入和输出文件夹路径")
    # 添加命令行参数
    parser.add_argument('-s', '--state', type=str, default="train",
                        help=r' 2状态：训练或推断')
    parser.add_argument('-t', '--train_csv', type=str, default='/home/ccwydlq10/code/Our/24.9.25/dataset.csv',
                        help=r'训练数据集的CSV文件路径')
    parser.add_argument('-v', '--val', type=str, default="/home/ccwydlq10/code/dataset/MEF/Lytro",
                        help=r'验证数据集的文件夹路径')
    parser.add_argument('-te', '--test', type=str, default="/home/ccwydlq10/code/dataset/MEF/MFFW/MFFW2",
                        help=r'测试数据集的文件夹路径')
    parser.add_argument('-st', '--step2', type=bool, default=False,
                        help=r'是否进入第二阶段')
    parser.add_argument('-m', '--model', type=str, default='CCNet',
                        help=r'选择训练的模型')
    parser.add_argument('-w', '--weights', type=str, default='_model/cp_20_14000_0.0185030996799469.pth',
                        help=r'GEU-Net模型权重的路径')
    parser.add_argument('-btr', '--batchsize_train', type=int, default=16,
                        help=r'训练时同时输入GPU的图像数量')
    parser.add_argument('-bv', '--batchsize_valid', type=int, default=16,
                        help=r'验证时同时输入GPU的图像数量')
    parser.add_argument('-bte', '--batchsize_test', type=int, default=1,
                        help=r'测试时同时输入GPU的图像数量')
    parser.add_argument('-g', '--gpus', type=str, default="0",
                        help=r'用于二值化的GPU数量')
    parser.add_argument('-r', '--lr', type=float, default=0.0005,
                        help=r'学习率')
    parser.add_argument('-e', '--epoch', type=int, default=20,
                        help=r'训练的轮数')
    parser.add_argument('-w1', '--w1_bce', type=int, default=1,
                        help=r'BCELoss的权重')
    parser.add_argument('-w2', '--w2_per', type=int, default=1,
                        help=r'感知损失(PerceptualLoss)的权重')
    parser.add_argument("-mi", "--model_inchannel", type=int, default=2,
                        help=r'模型输入通道数')
    parser.add_argument("-mo", "--model_outchannel", type=int, default=2,
                        help=r'模型输出通道数')
    return parser.parse_args()  # 返回解析后的命令行参数

class PerceptualLoss(nn.Module):
    def __init__(self, is_cuda):
        super(PerceptualLoss, self).__init__()
        print('加载resnet101模型...')
        # 初始化感知损失网络，使用预训练的ResNet101模型
        self.loss_network = models.resnet101(pretrained=True, num_classes=2, in_channel=2)
        # 关闭梯度计算以提高效率
        for param in self.loss_network.parameters():
            param.requires_grad = False
        if is_cuda:
            self.loss_network.cuda()  # 将模型移动到GPU
        print("完成模型加载...")

    def mse_loss(self, input, target):
        # 计算均方误差损失
        return torch.sum((input - target) ** 2) / input.data.nelement()

    def forward(self, output, label):
        # 前向传播计算感知损失
        self.perceptualLoss = self.mse_loss(self.loss_network(output), self.loss_network(label))
        return self.perceptualLoss

def import_model(model_input, in_channel=3, out_channel=1):
    # 根据输入模型的名称动态导入模型
    model_test = eval(model_input)(in_channel, out_channel)
    return model_test

def setting_cuda(gpus, model):
    # 设置CUDA设备
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # 设置可见的GPU设备
        gpu_list = gpus if type(gpus) is list else gpus.split(",")  # 将gpu字符串转为列表
        model.cuda()  # 将模型移动到GPU
        # 如果使用多个GPU，进行数据并行
        if len(gpu_list) > 1:
            model = torch.nn.DataParallel(model)

        print("使用GPU:", gpu_list, "进行训练。")
    else:
        gpu_list = []

    return gpu_list, model  # 返回GPU列表和模型
