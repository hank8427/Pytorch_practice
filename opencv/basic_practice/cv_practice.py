import cv2
import numpy as np
# img = cv2.imread("D:\Py_file\opencv\\allwant.jpg")
# img = cv2.resize(img,(450,250))
# new_img = img[:200, :100, :]
# img[50:250, 350:450, :] = new_img
# cv2.imshow('I want All',img)
# cv2.waitKey(10000)

vc = cv2.VideoCapture(0)
kernal = np.ones((5,5),np.uint8)
while True:
    ret, frame = vc.read()
    if ret:
        # frame = cv2.Canny(frame,50,100)
        cv2.imshow("video",frame)
    else:
        break
    if cv2.waitKey(50) == ord('z'):
        break

# img = np.empty((200,400,3),np.uint8)
# for row in range(200):
#     for col in range(400):
#         img[row][col] = [np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)]
# # print(img)
# cv2.imshow('img',img)
# cv2.waitKey(2000)