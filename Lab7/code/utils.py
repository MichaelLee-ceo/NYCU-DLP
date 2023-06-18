import os
import json
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid, save_image

def plot_result(losses, test_acc, new_test_acc, args):
    epoch_full = np.arange(0, args.num_epochs)
    epoch_sub = np.arange(0, args.num_epochs, 5)
    epoch_sub = np.append(epoch_sub, epoch_full[-1])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plt.title('Training loss / Accuracy curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epoch_full, losses, c='silver', label='mse')
    ax1.legend(loc='lower right')

    ax2.set_ylabel('Accuracy')
    ax2.plot(epoch_sub, test_acc, label='Test')
    ax2.plot(epoch_sub, new_test_acc, label='New Test')
    ax2.legend(loc='center right')

    fig.tight_layout()
    plt.savefig('{}/trainingCurve.png'.format(args.figure_dir))
    print("-- Save training figure")


def get_test_label(args, test_file):
    # 先讀 label 的對應檔, "gray cube" -> 0
    label_dict = json.load(open(os.path.join(args.dataset_dir, "objects.json")))
    labels = json.load(open(os.path.join(args.dataset_dir, test_file + ".json")))

    newLabels = []
    for i in range(len(labels)):
        onehot_label = torch.zeros(24, dtype=torch.float32)
        for j in range(len(labels[i])):
            onehot_label[label_dict[labels[i][j]]] = 1 
        newLabels.append(onehot_label)

    return newLabels

def evaluate(model, scheduler, epoch, args, device, test_file):
    test_label = torch.stack(get_test_label(args, test_file)).to(device)
    num_samples = len(test_label)

    x = torch.randn(num_samples, 3, args.sample_size, args.sample_size).to(device)
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_residual = model(x, t, test_label).sample

        x = scheduler.step(noise_residual, t, x).prev_sample

    image = (x / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    # image = (image * 255).round().astype("uint8")
    # pilImage = Image.fromarray(image)
    # pilImage.save("./figures/result.png")

    save_image(make_grid(image, nrow=8), "{}/{}_{}.png".format(args.figure_dir, test_file, epoch))
    # print('-- Save {}/{}_{}.png'.format(args.figure_dir, test_file, epoch))

    return x, test_label

def make_gif(model, scheduler, args, device, test_file):
    label = torch.stack(get_test_label(args, test_file)).to(device)
    num_samples = len(label)
    model.eval()

    frames = []
    x = torch.randn(num_samples, 3, args.sample_size, args.sample_size).to(device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_residual = model(x, t, label).sample
        x = scheduler.step(noise_residual, t, x).prev_sample

        image = (x / 2 + 0.5).clamp(0, 1)
        grid = make_grid(image, nrow=8).cpu().permute(1, 2, 0).numpy()
        grid = (grid * 255).round().astype("uint8")
        frames.append(grid)

    # store the gif
    gif_path = "{}/{}.gif".format(args.gif_dir, test_file)
    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(len(scheduler.timesteps) // 3):
                    writer.append_data(frames[-1])

    print("> Save {}".format(gif_path))