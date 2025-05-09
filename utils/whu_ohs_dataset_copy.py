import torch
from torch.utils import data
import os
import numpy as np
from osgeo import gdal
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

class WHU_OHS_Patch_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list, patch_size=17, data_aug=True, use_3D_input=False, channel_last=False):
        """
        :param image_file_list: List of image file paths
        :param label_file_list: List of corresponding label file paths
        :param patch_size: Size of the patch (e.g., 32)
        :param data_aug: Whether to use data augmentation, default is True
        :param use_3D_input: Whether to use 3D input
        :param channel_last: Whether the input channels are last (HWC format)
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2  # padding size (half patch size)
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last

        # Load images and labels
        self.images, self.labels = self._load_images_and_labels(image_file_list, label_file_list)

        # Prepare indices for patches centered on label values not equal to 0
        self.indices = self._prepare_patch_indices()

    def _load_images_and_labels(self, image_file_list, label_file_list):
        images = []
        labels = []
        for image_file, label_file in zip(image_file_list, label_file_list):
            image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
            label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)
            image = image_dataset.ReadAsArray()
            label = label_dataset.ReadAsArray()

            if self.channel_last:
                image = image.transpose(1, 2, 0)

            # Pad the image and label with reflect mode padding
            padded_image = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
            padded_label = np.pad(label, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

            images.append(padded_image)
            labels.append(padded_label)

        return images, labels

    def _prepare_patch_indices(self):
        indices = []
        for idx, label in enumerate(self.labels):
            # Create a mask where label values are not zero
            mask = label[self.ps:-self.ps, self.ps:-self.ps] > 0  # exclude padding area
            x_pos, y_pos = np.nonzero(mask)

            # Ensure the patch center is within valid bounds
            for x, y in zip(x_pos, y_pos):
                center_x, center_y = x + self.ps, y + self.ps  # adjust for padding offset
                indices.append((idx, center_x, center_y))

        np.random.shuffle(indices)  # Shuffle indices to introduce randomness
        return indices

    def hsi_augment(self, data):
        """ Perform data augmentation with random flip or rotation """
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image_idx, center_x, center_y = self.indices[i]

        # Retrieve the corresponding padded image and label
        image = self.images[image_idx]
        label = self.labels[image_idx]

        # Define the patch bounds
        x1, y1 = center_x - self.ps, center_y - self.ps
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # Extract the patch
        patch = image[x1:x2, y1:y2]
        patch_label = label[center_x, center_y] - 1  # Adjust label to be zero-based
        print(f"x1:{x1}, x2:{x2}, y1:{y1}, y2:{y2}")
        print(patch.shape)
        sleep(10)
        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            patch = self.hsi_augment(patch)

        # Convert patch to PyTorch tensor
        patch = np.asarray(np.copy(patch).transpose((2, 0, 1)), dtype='float32')
        patch = torch.from_numpy(patch)

        # Convert label to PyTorch tensor
        patch_label = torch.tensor(patch_label, dtype=torch.long)

        if self.use_3D_input:
            patch = patch.unsqueeze(0)

        return patch, patch_label


def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
    
def compute_HSI(y_test, y_pred_test):

    classification = classification_report(y_test, y_pred_test, digits=4)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa