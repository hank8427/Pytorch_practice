import cv2
import numpy as np

def empty(v):
    pass

# img = cv2.imread("D:\Py_file\opencv\\allwant.jpg")
# img = cv2.resize(img, (600,400))


vc = cv2.VideoCapture(0)

# 控制條
cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 600, 400)
                                    #  default, Max 
cv2.createTrackbar('Hue Min', 'TrackBar', 0 , 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBar', 179 , 179, empty)
cv2.createTrackbar('Saturation Min', 'TrackBar', 0 , 255, empty)
cv2.createTrackbar('Saturation Max', 'TrackBar', 255 , 255, empty)
cv2.createTrackbar('Value Min', 'TrackBar', 0 , 255, empty)
cv2.createTrackbar('Value Max', 'TrackBar', 255 , 255, empty)

while True:
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBar')
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBar')
    s_min = cv2.getTrackbarPos('Saturation Min', 'TrackBar')
    s_max = cv2.getTrackbarPos('Saturation Max', 'TrackBar')
    v_min = cv2.getTrackbarPos('Value Min', 'TrackBar')
    v_max = cv2.getTrackbarPos('Value Max', 'TrackBar')
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    ret, img = vc.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_or(img, img, mask = mask)
    
    cv2.imshow('img', img)
    # cv2.imshow('hsv', hsv)
    cv2.imshow('result', result)
    cv2.imshow('mask', mask)
    cv2.waitKey(50)



