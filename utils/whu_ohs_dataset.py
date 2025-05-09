import torch
from torch.utils import data
import os
import numpy as np
from osgeo import gdal
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class WHU_OHS_Patch_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list, patch_size=17, data_aug=False, use_3D_input=False, channel_last=False):
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
        self.ps = self.patch_size // 2  # padding size
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last

        # Load images and labels
        self.images, self.labels = self._load_images_and_labels(image_file_list, label_file_list)

        # Prepare indices for patches centered on label values not equal to 0
        #self.indices = self._prepare_patch_indices()
        self.indices, self.class_counts = self._prepare_patch_indices_class_counts()

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
            padded_image = np.pad(image, ((0, 0), (self.ps, self.ps), (self.ps, self.ps)), mode='reflect')
            padded_label = np.pad(label, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

            images.append(padded_image)
            labels.append(padded_label)
            #if padded_image.shape[0] != 32 or padded_image.shape[1] != 526:
            #    raise ValueError("do not match")

        return images, labels

    def _prepare_patch_indices(self):
        indices = []
        for idx, label in enumerate(self.labels):
            # Create a mask where label values are not zero
            mask = label > 0
            x_pos, y_pos = np.nonzero(mask)

            # Ensure the patch center is within valid bounds
            for x, y in zip(x_pos, y_pos):
                if self.ps <= x < label.shape[0] - self.ps and self.ps <= y < label.shape[1] - self.ps:
                    indices.append((idx, x, y))

        np.random.shuffle(indices)  # Shuffle indices to introduce randomness
        return indices

    def _prepare_patch_indices_class_counts(self):
        indices = []
        class_counts = {}
        for idx, label in enumerate(self.labels):
            # Create a mask where label values are not zero
            mask = label > 0
            x_pos, y_pos = np.nonzero(mask)

            # Ensure the patch center is within valid bounds
            for x, y in zip(x_pos, y_pos):
                if self.ps <= x < label.shape[0] - self.ps and self.ps <= y < label.shape[1] - self.ps:
                    indices.append((idx, x, y))
                    class_label = label[x, y]
                    if class_label in class_counts:
                        class_counts[class_label] += 1
                    else:
                        class_counts[class_label] = 1

        np.random.shuffle(indices)  # Shuffle indices to introduce randomness
        return indices, class_counts
    
    def hsi_augment(self, data):
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                # Flip left-right (horizontal flip) for C*H*W
                data = np.flip(data, axis=2)
            elif 0.2 < prob <= 0.4:
                # Flip up-down (vertical flip) for C*H*W
                data = np.flip(data, axis=1)
            elif 0.4 < prob <= 0.6:
                # Rotate 90 degrees clockwise (equivalent to k=-1 for C*H*W)
                data = np.rot90(data, k=1, axes=(1, 2))
            elif 0.6 < prob <= 0.8:
                # Rotate 180 degrees
                data = np.rot90(data, k=2, axes=(1, 2))
            elif 0.8 < prob <= 1.0:
                # Rotate 270 degrees clockwise (equivalent to k=-3 for C*H*W)
                data = np.rot90(data, k=3, axes=(1, 2))
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
        patch = image[:, x1:x2, y1:y2]
        patch_label = label[center_x, center_y] - 1  # Adjust label to be zero-based

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            patch = self.hsi_augment(patch)

        # Convert patch to PyTorch tensor
        patch = np.asarray(np.copy(patch), dtype='float32')
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

if __name__ == "__main__":
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
        
    print("train image number: ", len(train_image_list))
    print("val image number: ", len(val_image_list))
    print("test image number: ", len(test_image_list))
    
    train_dataset = WHU_OHS_Patch_Dataset(image_file_list=train_image_list, label_file_list=train_label_list, 
                                      patch_size=15, data_aug=False, use_3D_input=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8,
                                            pin_memory=True)
    print("train data ready!")
    print("train data number: ", len(train_loader.dataset))
    print("train data class counts:")
    for class_label, count in train_dataset.class_counts.items():
        print(f"Class {class_label}: {count} samples")

    val_dataset = WHU_OHS_Patch_Dataset(image_file_list=val_image_list, label_file_list=val_label_list, 
                                        patch_size=15, data_aug=False, use_3D_input=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("val data ready!")
    print("val data number: ", len(val_loader.dataset))
    print("val data class counts:")
    for class_label, count in val_dataset.class_counts.items():
        print(f"Class {class_label}: {count} samples")

    test_dataset = WHU_OHS_Patch_Dataset(image_file_list=test_image_list, label_file_list=test_label_list, 
                                        patch_size=16, data_aug=False, use_3D_input=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("test data ready!")
    print("test data number: ", len(test_loader.dataset))
    print("test data class counts:")
    for class_label, count in test_dataset.class_counts.items():
        print(f"Class {class_label}: {count} samples")
