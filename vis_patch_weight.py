import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
from models.get_model import get_model
import json
from utils.dataset import load_mat_hsi, HSIDataset, HSI_gt_Dataset, HSIDataset_vis
from tqdm import tqdm

device = torch.device("cuda:{}".format(2) if torch.cuda.is_available() else "cpu")
def list_to_colormap(x_list, name):
    y = np.zeros((x_list.shape[0], 3))
    if name != 'hu2018':
        for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([0, 0, 0]) / 255.
            if item == 1:
                y[index] = np.array([255, 0, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 3:
                y[index] = np.array([0, 0, 255]) / 255.
            if item == 4:
                y[index] = np.array([255, 255, 0]) / 255.
            if item == 5:
                y[index] = np.array([0, 255, 255]) / 255.
            if item == 6:
                y[index] = np.array([255, 0, 255]) / 255.
            if item == 7:
                y[index] = np.array([192, 192, 192]) / 255.
            if item == 8:
                y[index] = np.array([128, 128, 128]) / 255.
            if item == 9:
                y[index] = np.array([128, 0, 0]) / 255.
            if item == 10:
                y[index] = np.array([128, 128, 0]) / 255.
            if item == 11:
                y[index] = np.array([0, 128, 0]) / 255.
            if item == 12:
                y[index] = np.array([128, 0, 128]) / 255.
            if item == 13:
                y[index] = np.array([0, 128, 128]) / 255.
            if item == 14:
                y[index] = np.array([0, 0, 128]) / 255.
            if item == 15:
                y[index] = np.array([255, 165, 0]) / 255.
            if item == 16:
                y[index] = np.array([255, 215, 0]) / 255.
    return y

def visualize_features_and_gt(model, data_loader, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置模型为评估模式
    model.eval()
    model.to(device)

    # 遍历数据和ground truth数据
    for idx, (data, gt) in tqdm(enumerate(data_loader), total=len(data_loader), desc='vis_patch_features'):
        data = data.to(device)

        with torch.no_grad():
            # 前向传播获取特征和注意力权重
            _ = model(data)

            # 提取特征和注意力权重
            #x_ssm_features = model.x_ssm_outputs
            #original_ssm_spa_featurs = model.original_ssm_spa_output
            attn_spa_weights = model.attn_spa_outputs
            attn_spe_weights = model.attn_spe_outputs
            


        # 可视化注意力权重热力图
        attn_spa = attn_spa_weights[0].cpu().numpy()[0, 0, :]  # 提取第一个 block 的 attn_spa
        attn_spe = attn_spe_weights[0].cpu().numpy()[0, 0, :]  # 提取第一个 block 的 attn_spe

        # 标准化注意力权重到0-1
        attn_spa = (attn_spa - attn_spa.min()) / (attn_spa.max() - attn_spa.min())
        attn_spe = (attn_spe - attn_spe.min()) / (attn_spe.max() - attn_spe.min())

        # 保存注意力权重热力图
        attn_spa_path = os.path.join(output_dir, f"{idx}_a_heatmap.png")
        attn_spe_path = os.path.join(output_dir, f"{idx}_e_heatmap.png")

        # 可视化和保存
        plt.figure(figsize=(12, 1))  # 调整图像比例，使其成为长条形
        plt.imshow(attn_spa.reshape(1, -1), aspect='auto', cmap='jet')
        plt.axis('off')  # 去掉坐标轴
        plt.colorbar()
        plt.title('Spatial Attention Weight')
        plt.savefig(attn_spa_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure(figsize=(12, 1))  # 调整图像比例，使其成为长条形
        plt.imshow(attn_spe.reshape(1, -1), aspect='auto', cmap='jet')
        plt.axis('off')  # 去掉坐标轴
        plt.colorbar()
        plt.title('Spectral Attention Weight')
        plt.savefig(attn_spe_path, bbox_inches='tight', pad_inches=0)
        plt.close()


# 加载模型
config_dir = os.path.join('config', '%s_%s.json' % ('asf_rsm_group', 'ip'))
if os.path.exists(config_dir):
    model_config = json.load(open(config_dir, 'r'))

model = get_model(model_config['model'], 'asf_rsm_group', 'ip', 7)  # 请使用您的模型构造函数
best_model_dir = "/remote-home/share/shengjiamu/Group-Aware-Hierarchical-Transformer/checkpoints/asf_rsm_group/ip/0.1/2024-05-25_16-52-07/0/model_best.pth"  # 训练好的模型权重路径
model.load_state_dict(torch.load(best_model_dir, map_location=device))

# 数据加载器
image, gt, labels = load_mat_hsi('ip', './datasets')
height, width = gt.shape
num_classes = len(labels)
num_bands = image.shape[-1]
all_data_set = HSIDataset_vis(image, gt, patch_size=7, data_aug=False)
all_data_loader = torch.utils.data.DataLoader(all_data_set, 1, num_workers=1, drop_last=False, shuffle=False)
#all_gt_set = HSI_gt_Dataset(gt, patch_size=7)
#all_gt_loader = torch.utils.data.DataLoader(all_gt_set, 1, num_workers=1, drop_last=False, shuffle=False)

# 可视化输出目录
output_dir = "./vis_patch_weight"

# 调用可视化函数
visualize_features_and_gt(model, all_data_loader, output_dir)
