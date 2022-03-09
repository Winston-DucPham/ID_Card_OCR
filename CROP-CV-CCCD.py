import cv2
from matplotlib import pyplot as plt
import numpy as np


im = cv2.imread('train/images597493_cmnd-6.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('cccd.jpg', 0)
kernel = np.ones((15,15),np.uint8)
blur = cv2.GaussianBlur(img_gray,(5,5),0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
cl1 = clahe.apply(blur)
edges = cv2.Canny(cl1,50,100)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
ret2, otsu = cv2.threshold(closing,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.drawContours(im, contours, -1, (0,255,0), 6)

contours, hierarchy = cv2.findContours(otsu,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
    area = cv2.contourArea(cnt)

    if area > 40000:

      box = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 3) 

      cv2.putText(box, 'box', (x+50,y+50),cv2.FONT_HERSHEY_SIMPLEX, 1.5 ,(0,255,255), 2)


plt.figure(figsize=(15,8))
plt.imshow(im, "gray")
plt.show()