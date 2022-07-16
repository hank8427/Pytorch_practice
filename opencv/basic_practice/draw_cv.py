import cv2
import numpy as np

img = np.zeros((600, 600, 3), np.uint8)
cv2.line(img, (0,0), (img.shape[1],img.shape[0]), (255,0,255), 3)
cv2.rectangle(img, (0,100), (400,300), (0,0,255), cv2.FILLED)
cv2.circle(img, (500,500), 30, (255,0,0), 3)
                            # 左下角的座標    字體            字體大小   顏色      粗度
cv2.putText(img, 'ALL want' , (0,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)

cv2.imshow('img',img)
cv2.waitKey(1000)