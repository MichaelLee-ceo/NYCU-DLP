import os
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import mkdir

mkdir('./data/train')
mkdir('./data/test')

datasets = ['train', 'test']
for dataset in datasets:
    image_paths = pd.read_csv(dataset + "_img.csv", header=None)
    image_paths = np.squeeze(image_paths)           # 拿掉最外層的 dimension

    print('\n----- Preprocessing on {} data: {} -----'.format(dataset, len(image_paths)))
    for path in tqdm(image_paths):
        img_path = os.getcwd() + "/data/new_" + dataset + "/" + path + '.jpeg'
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        width, height = img.shape[1], img.shape[0]
        new_width, new_height = 0, 0

        # 找出比較短的那一邊，等比例做 downsample / upsample 成最小邊是 512
        shorter_side = min(width, height)
        ratio = 512 / shorter_side
        new_width, new_height = math.ceil(width*ratio), math.ceil(height*ratio)

        new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # applies center crop
        center = new_img.shape
        w, h = 512, 512
        x = int(center[1] // 2 - 256)
        y = int(center[0] // 2 - 256)
        crop_img = new_img[y:y+h, x:x+w]

        cv2.imwrite("./data/" + dataset + "/" + path + '.jpeg', crop_img)
        if crop_img.shape[0] != 512 or crop_img.shape[1] != 512:
            print('[{}] original size: ({}, {}), resized: ({}, {})'.format(dataset, width, height, crop_img.shape[1], crop_img.shape[0]))
            print(path)
            print(new_img.shape, crop_img.shape, x, y)