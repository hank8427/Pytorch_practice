import cv2
import numpy as np
img = cv2.imread("allwant.jpg")

# Convert colorful image to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# Fuzzy the image (core function size, std)
fuzzy = cv2.GaussianBlur(img, (5,5), 3)
cv2.imshow('fuzzy',fuzzy)

# Edge detection (img, low thr, high thr)
canny = cv2.Canny(img, 150, 150)
cv2.imshow('canny',canny)

# dilate膨脹 (img, kernal, iterations)
kernal = np.ones((3,3), np.uint8)
dilate = cv2.dilate(canny, kernal, iterations = 1)
cv2.imshow('dilate',dilate)

# erode侵蝕 (img, kernal, iterations)
kernal = np.ones((3,3), np.uint8)
erode = cv2.erode(dilate, kernal, iterations = 1)
cv2.imshow('erode',erode)
cv2.waitKey(0)