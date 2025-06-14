import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import grouper, sliding_window, count_sliding_window


def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):
    best_acc = -0.1
    losses = []
    break_count = 0
    for e in tqdm(range(1, epoch+1), desc="training the network"):
        network.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, epoch,  mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        val_acc = validation(network, val_loader, device)
        #val_acc = 0
        if scheduler is not None:
            scheduler.step()
        if val_acc >= best_acc:
            is_best = 1
            best_acc = val_acc
            save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)
            break_count = 0
        else:
            break_count += 1
            if break_count > 1000:
                break
        
    

def validation(network, val_loader, device):
    num_correct = 0.
    total_num = 0.
    network.eval()
    for batch_idx, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = network(images)
        _, outputs = torch.max(outputs, dim=1) 
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            total_num = total_num + 1
    overall_acc = num_correct / total_num
    return overall_acc


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()
    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best acc = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    #else:  # save the ckpt for each 10 epoch
        #if kwargs['epoch'] % 10 == 0:
        #    torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))



