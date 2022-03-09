import os
import cv2
import re
import pandas as pd
from modules import retrieval, correction
from modules.ocr.tool import predictor
#from tool.utils import natural_keys, visualize
import time
import matplotlib.pyplot as plt

class_mapping = {"ID_Num":0, "Name":1, "DOB":2, "Sex":3, "Country":4,"Address_orgin":5,"Address_Usual":6,"Valid_date":7}
idx_mapping = {0:"ID_Num", 1:"Name", 2:"DOB", 3:"Sex", 4:"Country",5:"Address_orgin",6:"Address_Usual",7:"Valid_date"}
ocr_weight = 'weights/transformerocr.pth'
ocr_model = predictor(ocr_weight)

#retrieval = retrieval(class_mapping, mode = 'all')
correction = correction()
