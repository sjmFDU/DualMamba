import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
import os
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset, load_disjoint_data
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
import utils.get_cls_map as get_cls_map
from models.get_model import get_model
from train import train, test
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from tqdm import tqdm
'''
color = [
    'yellowgreen','yellow','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow',
    'red','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown','burlywood'
]'''
color = ['indianred', 'red', 'chocolate', 'orange', 'yellowgreen', 'lime', 'forestgreen', 'lightseagreen', 'deepskyblue', 'slategray', 'royalblue', 'blue', 'mediumorchid', 'purple', 'fuchsia', 'pink']
class_name = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    #data = np.exp(data) / np.sum(np.exp(data), axis = 1, keepdims = True)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    colorlist = color

    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        colorplt = color[int(label[i])]
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        #          fontdict={'weight': 'bold', 'size': 7})
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=colorplt,
        #          fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], s=5, color=colorplt, marker='.', label=label[i])
    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    if title:
        plt.title(title, fontsize=14)
    # 返回值
    return fig

if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='cnn3d')
    parser.add_argument("--dataset_name", type=str, default="ip")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--num_run", type=int, default=1) 
    parser.add_argument("--epoch", type=int, default=300)    
    parser.add_argument("--bs", type=int, default=64)  # bs = batch size  
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--disjoint", type=bool, default=False)
    opts = parser.parse_args()
    config_dir = os.path.join('config', '%s_%s.json' % (opts.model, opts.dataset_name))
    import json
    if os.path.exists(config_dir):
        model_config = json.load(open(config_dir, 'r'))
    else:
        model_config = {}
    device = torch.device("cuda:{}".format(opts.device))
    if opts.model == 'ssftt':
        if opts.dataset_name == 'hu2018':
            opts.patch_size = 9
        else:
            opts.patch_size = 13
    elif opts.model == 'speformer':
        opts.patch_size = 7
    elif opts.model == 'cnn3d':
        opts.patch_size = 13
    elif opts.model == 'cnn2d':
        opts.patch_size = 13
    elif opts.model == 'ssrn':
        opts.patch_size = 9
    elif opts.model == 'posfree_vit':
        opts.patch_size = 7
    elif opts.model == 'asf_group':
        opts.patch_size = 9
    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training and {} testing".format(opts.ratio, 1 - opts.ratio))

    # load data
    

    # random seeds
    seeds = [202202, 202203, 202204, 202205, 23, 424124, 231419123, 21213, 123, 29]

    # empty list to storing results
    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run+1, opts.num_run))
        image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
        height, width = gt.shape
        num_classes = len(labels)
        num_bands = image.shape[-1]
        # get train_gt, val_gt and test_gt
        '''trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt'''
        if opts.disjoint:
            train_gt, test_gt = load_disjoint_data(opts.dataset_name, opts.dataset_dir)
            train_gt -= 1
            print(len(np.where(train_gt > -1)[0]))
            if opts.dataset_name == 'ip':
                import copy
                all_gt = copy.deepcopy(gt)
                train_gt[all_gt == -1] = -1
                all_gt[train_gt != -1] = -1
                test_gt = all_gt
                del all_gt
            elif opts.dataset_name == 'pu':
                train_gt[gt == -1] = -1
                test_gt -= 1
            print(len(np.where(train_gt > -1)[0]))
        else: 
            train_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        val_gt = test_gt
        #val_gt, _ = sample_gt(test_gt, 0.5, seeds[run])
        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
        all_data_set = HSIDataset(image, gt, patch_size=opts.patch_size, data_aug=False)
        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        all_data_loader = torch.utils.data.DataLoader(all_data_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        # load model and loss
        model = get_model(model_config['model'], opts.model, opts.dataset_name, opts.patch_size)
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(opts.ratio) + '/' + str(run)
        model.load_state_dict(torch.load(model_dir + "/model_best.pth"))
        model = model.to(device)
        model.eval()
        all_features = []
        all_labels = []
        for image, target in tqdm(all_data_loader):
            image = image.to(device)
            with torch.no_grad():
                feature = model(image)
                all_features.append(feature.cpu().numpy())
                all_labels.append(target.numpy())
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        ts = TSNE(n_components=2, random_state=0)
        result = ts.fit_transform(all_features)
        fig = plot_embedding(result, all_labels, title = None)
        plt.savefig("tsne_{}_{}.png".format(opts.model, opts.dataset_name))
