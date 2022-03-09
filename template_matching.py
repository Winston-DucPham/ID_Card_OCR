from PIL import Image
import pandas as pd
import cv2
import os

def resize_img_as_template(img):
    resized_img = cv2.resize(img, (500, 375), interpolation=cv2.INTER_AREA)
    return resized_img


def read_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def draw_box_as_template(dataframe, img):
   # img = resize_img_as_template(imgpath)
    for i in range(0, 10):
        img = cv2.rectangle(img, (dataframe.xmin[i], dataframe.ymin[i]), (dataframe.xmax[i], dataframe.ymax[i]),
                            (0, 255, 0), 2)
    return img



def read_image_with_coordinate(dataframe, img):
    #img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
   # img = resize_img_as_template(img)
    dir= 'resultimgdet/'
    if len(os.listdir(dir)) != 0:
        for i in os.listdir(dir):
            os.remove(os.path.join(dir, i))

    for i in range(0,10):
        #img[y:h,x:w]
        crop_box = img[dataframe.ymin[i]:(dataframe.ymax[i]),
                   dataframe.xmin[i]:(dataframe.xmax[i])]
        savepath = os.path.join(dir,'box_{}.jpg'.format(i))
        cv2.imwrite(savepath, crop_box)




