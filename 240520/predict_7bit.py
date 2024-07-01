import numpy as np
import cv2
import os
from glob import glob
from os import path as osp
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import math

# ansung
foldername = 'hadong'

DIR = f'/workspace/datasets/pig/7bit/src/{foldername}'

SAVE_DIR = osp.join(DIR.replace(f'/{foldername}',f'/{foldername}_7bit'), foldername)
HIST_SAVE_DIR = osp.join(DIR.replace(f'/{foldername}',f'/{foldername}_7bit'), 'hist_7bit')

def linear_map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def save_histogram(img, hist_file_path):
    # Compute histogram
    hist = cv2.calcHist([img], [0], None, [128], [0, 128])
    # Plot histogram
    plt.plot(hist)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    # Save the histogram
    plt.savefig(hist_file_path)
    plt.clf()
    

def log_transform(pixel_value):
    return np.uint8(255 * (np.log(1 + pixel_value) / np.log(1 + 255)))


def main():
    if not osp.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    if not osp.exists(HIST_SAVE_DIR):
        os.makedirs(HIST_SAVE_DIR)

    files = sorted(glob(osp.join(DIR, "*g")))
    
    for file in files:
        fname = osp.basename(file)

        img = cv2.imread(file, 0)

        # proposed method
        ###################
            
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                img[i, j] = log_transform(img[i, j])

        min_pixel = np.min(img)
        max_pixel = np.max(img)

        input_min = min_pixel
        input_max = max_pixel
        output_min = 64
        output_max = 191

        for i in range(rows):
            for j in range(cols):
                pixel_value = img[i, j]
                img[i, j] = np.clip(linear_map(pixel_value, input_min, input_max, output_min, output_max), output_min, output_max)


        # simple method
        ###########
        # img = (img >> 1).astype(np.uint8) + 64
        

        save_path = osp.join(SAVE_DIR, fname)
        
        cv2.imwrite(save_path, img)

        hist_file_path = osp.join(HIST_SAVE_DIR, f"{fname}.png")
        save_histogram(img, hist_file_path)

 
if __name__ == '__main__':
    main()