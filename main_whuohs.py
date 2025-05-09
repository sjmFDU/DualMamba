import logging
import sys
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
#from torchsummaryX import summary
import os
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset, load_disjoint_data
from utils.whu_ohs_dataset import WHU_OHS_Patch_Dataset, compute_HSI, test
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
import utils.get_cls_map as get_cls_map
from models.get_model import get_model
from train import train
import json
import datetime
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_filename='output.log'):
    """设置日志记录，同时记录到文件和控制台。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename=log_filename,
        filemode='a'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console_handler)



if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='ssmamba')
    parser.add_argument("--dataset_name", type=str, default="whu-ohs")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--patch_size", type=int, default=15)
    parser.add_argument("--num_run", type=int, default=1) 
    parser.add_argument("--epoch", type=int, default=1)    
    parser.add_argument("--bs", type=int, default=16)  # bs = batch size  

    opts = parser.parse_args()
    config_dir = os.path.join('config', '%s_%s.json' % (opts.model, opts.dataset_name))
    if os.path.exists(config_dir):
        model_config = json.load(open(config_dir, 'r'))
    else:
        model_config = {'model': '', 'schedule': ''}
    now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = '{}_{}_{}.log'.format(opts.model, opts.dataset_name, now_time)
    log_dir = os.path.join('logs', log_filename)
    setup_logging(log_dir)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    print("model_config:\n%s", json.dumps(model_config, indent=4))
    device = torch.device("cuda:{}".format(opts.device))
    if opts.model == 'ssftt':
        if opts.dataset_name == 'hu2018':
            opts.patch_size = 9
        else:
            opts.patch_size = 13
    elif opts.model == 'speformer':
        opts.patch_size = 13
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
    elif opts.model == "ssmamba":
        opts.patch_size = 27
    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))

    # rs_mamba seeds
    #seeds = [202202, 202203, 202204, 202205, 23, 424124, 231419123, 21213, 123, 29]
    if opts.dataset_name == 'ip':
        #seeds = [1, 34, 20031025, 9, 10, 11, 424124, 21213, 123, 29] #wo5.7 ip bad:202,202202,202203,202204,20000707,20000708, mid:12, good:1, 34, 20031025, 9, 10, 11, 424124, 21213, 123, 29
        #seeds = [202]
        seeds = [202202, 202203, 202204, 20000707, 20000708, 12, 202, 29, 21213, 5190]
    elif opts.dataset_name == 'whulk':
        seeds = [202205, 23, 231419123, 21213, 29]
        #seeds = [202202] #wo1.2.3 whulk, 424124,123
    elif opts.dataset_name == 'hu2018':
        seeds = [202203, 1, 34] #1 2 4 5 hu2018 good: 202202,202203,424124, bad:202205 
        #seeds = [1]
    # empty list to storing results
    results = []
    #common seed
    seeds = [1,2,3,4,5,6,7,8,9]
    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        torch.manual_seed(seeds[run])
        torch.cuda.manual_seed_all(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run+1, opts.num_run))
        '''
        image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
        height, width = gt.shape
        num_classes = len(labels)
        num_bands = image.shape[-1]
        # get train_gt, val_gt and test_gt
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt
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
        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        all_data_loader = torch.utils.data.DataLoader(all_data_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        '''

        # load model and loss
        model = get_model(model_config['model'], opts.model, opts.dataset_name, opts.patch_size)
        num_bands = 32
        '''if run == 0:
            split_info_print(train_gt, test_gt, labels)
            print("network information:")
            with torch.no_grad():
                summary(model, torch.zeros((1, num_bands, opts.patch_size, opts.patch_size)))'''
        
        model = model.to(device)
        

        #total_params = sum(p.numel() for p in model.parameters())
        if run == 0:
            for name, module in model.named_modules():
                # 计算每个模块的参数量
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                # 打印模块名称和参数量
                print(f"{name}: {num_params} parameters")
            #model.flops()
        #print("total parameters: {}".format(total_params))
        optimizer, scheduler = load_scheduler(model_config['schedule'], opts.model, model)

        criterion = nn.CrossEntropyLoss()
        
        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + now_time + '/'+ str(run)

        #load data
        print('Load data ...')

        data_root = '/remote-home/share/WHU-OHS'
        image_prefix = 'S1'
        data_path_train_image = os.path.join(data_root, 'tr', 'image')
        data_path_val_image = os.path.join(data_root, 'val', 'image')
        data_path_test_image = os.path.join(data_root, 'ts', 'image')

        train_image_list = []
        train_label_list = []
        val_image_list = []
        val_label_list = []
        test_image_list = []
        test_label_list = []

        for root, paths, fnames in sorted(os.walk(data_path_train_image)):
            for fname in fnames:
                if is_image_file(fname):
                    if ((image_prefix + '_') in fname):
                        image_path = os.path.join(data_path_train_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        train_image_list.append(image_path)
                        train_label_list.append(label_path)

        for root, paths, fnames in sorted(os.walk(data_path_val_image)):
            for fname in fnames:
                if is_image_file(fname):
                    if ((image_prefix + '_') in fname):
                        image_path = os.path.join(data_path_val_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        val_image_list.append(image_path)
                        val_label_list.append(label_path)
        
        for root, paths, fnames in sorted(os.walk(data_path_test_image)):
            for fname in fnames:
                if is_image_file(fname):
                    if ((image_prefix + '_') in fname):
                        image_path = os.path.join(data_path_test_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        test_image_list.append(image_path)
                        test_label_list.append(label_path)
        
        train_dataset = WHU_OHS_Patch_Dataset(image_file_list=train_image_list, label_file_list=train_label_list, 
                                      patch_size=opts.patch_size, data_aug=False, use_3D_input=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=8,
                                           pin_memory=True)
        print("train data ready!")
        val_dataset = WHU_OHS_Patch_Dataset(image_file_list=val_image_list, label_file_list=val_label_list, 
                                    patch_size=opts.patch_size, data_aug=False, use_3D_input=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts.bs, shuffle=False, num_workers=8, pin_memory=True)

        print("val data ready!")
        test_dataset = WHU_OHS_Patch_Dataset(image_file_list=test_image_list, label_file_list=test_label_list, 
                                    patch_size=opts.patch_size, data_aug=False, use_3D_input=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.bs, shuffle=False, num_workers=8, pin_memory=True)
        print("test data ready!")

        try:
            train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')

        # test the model
        y_pred_test, y_test = test(device, model, test_loader)
        classification, oa, confusion, each_acc, aa, kappa = compute_HSI(y_test, y_pred_test)
        print(classification)
        print("------------------")
        print(confusion)
        print(each_acc)
        print("------------------")
        text = ""
        text += "Accuracy : {:.02f}%\n".format(oa)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.04f}\n".format(kappa)
        print(text)



