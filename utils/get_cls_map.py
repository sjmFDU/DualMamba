import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y, height, width):
    k = 0
    
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

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
    if name == 'hu2018':        
        for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([0, 0, 0]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 65]) / 255.
            if item == 2:
                y[index] = np.array([37, 255, 59]) / 255.
            if item == 3:
                y[index] = np.array([0, 0, 255]) / 255.
            if item == 4:
                y[index] = np.array([0, 138, 86]) / 255.
            if item == 5:
                y[index] = np.array([0, 138, 35]) / 255.
            if item == 6:
                y[index] = np.array([164, 75, 155]) / 255.
            if item == 7:
                y[index] = np.array([0, 255, 255]) / 255.
            if item == 8:
                y[index] = np.array([242, 242, 242]) / 255.
            if item == 9:
                y[index] = np.array([226, 191, 211]) / 255.
            if item == 10:
                y[index] = np.array([255, 0, 0]) / 255.
            if item == 11:
                y[index] = np.array([175, 158, 141]) / 255.
            if item == 12:
                y[index] = np.array([128, 125, 120]) / 255.
            if item == 13:
                y[index] = np.array([182, 0, 0]) / 255.
            if item == 14:
                y[index] = np.array([90, 0, 0]) / 255.
            if item == 15:
                y[index] = np.array([248, 155, 0]) / 255.
            if item == 16:
                y[index] = np.array([255, 255, 37]) / 255.
            if item == 17:
                y[index] = np.array([255, 147, 0]) / 255.
            if item == 18:
                y[index] = np.array([255, 0, 249]) / 255.
            if item == 19:
                y[index] = np.array([0, 5, 253]) / 255.
            if item == 20:
                y[index] = np.array([172, 197, 220]) / 255.
    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def test(device, net, all_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    
    for inputs, labels in all_loader:
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

def get_cls_map(prediction, y, height, width, name, ratio, model_name):

    ignored_mask = np.zeros(y.shape[:2], dtype=bool)
    ignored_mask[y < 0] = True
    ignored_mask = ~ignored_mask
    #y = y[ignored_mask]
    prediction = prediction[ignored_mask]
    y += 1

    cls_labels = get_classification_map(prediction, y, height, width)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x, name)
    y_gt = list_to_colormap(gt, name)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    classification_map(y_re, y, 300,
                       './results/' + '{}_{}_predictions_{}.png'.format(model_name, name, ratio))
    #classification_map(gt_re, y, 300,
                       #'{}_gt.png'.format(name))
    print('------Get classification maps successful-------')
