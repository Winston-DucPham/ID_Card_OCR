import os
import cv2
import argparse
from template_matching import *
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from modules import OCR
from modules.preprocess.scan import *
from tool.config import Config
from tool.utils import natural_keys
from datetime import datetime
import timeit


parser = argparse.ArgumentParser("ID Card Extraction")
parser.add_argument("--debug", action="store_true", help="Save every steps for debugging")
args = parser.parse_args()


class Pipeline:
    def __init__(self, args, config):
        self.ocr_weight = config.ocr_weight
        self.output = 'outputtest'
        self.debug = args.debug
        self.load_config(config)
        self.make_cache_folder()
        self.init_modules()

    def load_config(self, config):
        self.ocr_config = config.ocr_config

    def make_cache_folder(self):
        self.final_output = os.path.join(self.output, 'result.jpg')

    def init_modules(self):
        self.ocr_model = OCR(
            config_path=self.ocr_config,
            weight_path=self.ocr_weight)


    def preproces(self, img):
        scanner = DocScanner("i")
        img1 = scanner.scan(img)
        return img1

    def start(self, img):
        img_paths = os.listdir(img)
        img_paths.sort(key=natural_keys)
        img_paths = [os.path.join(img, i) for i in img_paths]

        texts = self.ocr_model.predict_folder(img_paths, return_probs=False)
        return texts


if __name__ == "__main__":
    imgpath = 'train/fb369703b5df7f8126ce.jpg'
    folderpath = 'resultimgdet/'
    csvpath = 'train/_annotations.csv'
    dataframe = read_csv(csvpath)
    #
    config = Config('./tool/config/configs.yaml')
    pipeline = Pipeline(args, config)
    img1 = pipeline.preproces(imgpath)
    #cv2.imwrite("image1.jpg",img1)
    #cv2.waitKey(0)
    read_image_with_coordinate(dataframe, img1)
    #img1 = draw_box_as_template(dataframe,img1)
   # cv2.imwrite("image1.jpg", img1)

    start = timeit.default_timer()
    text = pipeline.start(folderpath)
    stop = timeit.default_timer()
    if text[6].isdigit() == True:
       text[6] = ""
    print(text)
    extract = {
        "số chứng minh" : text[0],
        "Tên": text[1],
        "Ngày Tháng Năm Sinh": text[2],
        "Giới tính": text[3],
        "Quốc tịch": text[4],
        "Quê Quán": text[5] +","+text[6],
        "Nơi Thường Trú": text[7]+","+text[8],
        "Hiệu lực đến": text[9],
        }
    for key in extract:
        print(key, ' : ',extract[key])
    print('time: ', stop - start)
