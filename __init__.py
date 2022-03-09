import os
import cv2
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from modules.preprocess import DocScanner
import modules.ocr as ocr
import modules.retrieval as retrieval
import modules.correction as correction
import modules.preprocess as preprocess
from tool.config import Config
from tool.utils import download_pretrained_weights

CACHE_DIR = '.cache'

class OCR:
    def __init__(self, config_path=None, weight_path=None, model_name=None):
        if config_path is None:
            config_path = 'tool/config/ocr/configs.yaml'
        config = Config(config_path)
        ocr_config = ocr.Config.load_config_from_name(config.model_name)
        ocr_config['cnn']['pretrained'] = False
        ocr_config['device'] = 'cuda:0'
        ocr_config['predictor']['beamsearch'] = False

        self.model_name = model_name
        if weight_path is None:
            if self.model_name is None:
                self.model_name = "transformerocr_default_vgg"
            tmp_path = os.path.join(CACHE_DIR, f'{self.model_name}.pth')
            download_pretrained_weights(self.model_name, cached=tmp_path)
            weight_path = tmp_path
        ocr_config['weights'] = weight_path
        self.model = ocr.Predictor(ocr_config)

    def __call__(self, img, return_prob=False):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.model.predict(img, return_prob)

    def predict_folder(self, img_paths, return_probs=False):
        texts = []
        if return_probs:
            probs = []
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            if return_probs:
                text, prob = self(img, True)
                texts.append(text)
                probs.append(prob)
            else:
                text = self(img, False)
                texts.append(text)

        if return_probs:
            return texts, probs
        else:
            return texts
