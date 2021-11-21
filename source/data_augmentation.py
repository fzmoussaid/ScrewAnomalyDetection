import numpy as np
import os
from os import listdir
from os.path import isfile, join
import io
import time
import glob
import cv2
from PIL import Image, ImageStat

alpha = 2.0
beta = 0.3

def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    fill_R = img[0][0][0]
    fill_G = img[0][0][1]
    fill_B = img[0][0][2]
    img = cv2.warpAffine(img, M, (w, h), borderValue=(int(fill_R),int(fill_G),int(fill_B)))
    return img

def image_brightness_change(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
        

def data_augmentation(dir_path):
    img_list = [_ for _ in os.listdir(dir_path) if _.endswith('png')]
    for img_name in img_list: 
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        # Create new images using rotation of angles (45, 90, 135, 180, 225, 270, 315)
        for i in range(1,8):
            angle = i*45
            rotated_img = rotation(img, angle)
            rotated_img_name = img_name.split('.')[0] + '_rotation_' + str(angle) + '.png'
            rotated_img_path = os.path.join(dir_path, rotated_img_name )
            cv2.imwrite(rotated_img_path, rotated_img)
        # Create new image using different brightness
        img = image_brightness_change(img)
        bright_img_name = img_name.split('.')[0] + '_brightness.png'
        bright_img_path = os.path.join(dir_path, bright_img_name )
        cv2.imwrite(bright_img_path, img) 

def check_duplicates(dir_path):                                 
    duplicate_files = []
    img_list = [_ for _ in os.listdir(dir_path) if _.endswith('png')]
    for file_org in img_list:
        if not file_org in duplicate_files:
            image_org = Image.open(os.path.join(dir_path, file_org))
            pix_mean1 = ImageStat.Stat(image_org).mean

            for file_check in img_list:
                if file_check != file_org:
                    image_check = Image.open(os.path.join(dir_path, file_check))
                    pix_mean2 = ImageStat.Stat(image_check).mean

                    if pix_mean1 == pix_mean2:
                        duplicate_files.append((file_org))
                        duplicate_files.append((file_check))

    print(list(dict.fromkeys(duplicate_files)))