import cv2
import numpy as np

# img = cv2.imread('d:/Py_file/opencv/mshape.png')
vc = cv2.VideoCapture(0)
while True:
    ret, img = vc.read()
    if ret:
        img = cv2.resize(img, (0,0), fx = 0.8, fy = 0.8)
        imgContour = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img, 200, 300)
        _, contours, herarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            #                          所有點全畫    顏色    粗度
            cv2.drawContours(img, cnt, -1, (255,0,0), 3)
            area = cv2.contourArea(cnt)
            if area > 120:
                #                       是否閉合
                peri = cv2.arcLength(cnt, True)
                # 近似物體形狀                    越小精度越高  是否閉合
                vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)
                corners = len(vertices)
                print(corners)
                # x,y 左上角座標  w寬度 h高度
                x,y,w,h = cv2.boundingRect(vertices)
                # 以矩形框出物體
                cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,0,255), 3)
                if corners == 3:
                    cv2.putText(imgContour, 'Triangle', (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                elif corners == 4:
                    cv2.putText(imgContour, 'Rectangle', (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                elif corners == 5:
                    cv2.putText(imgContour, 'Pentagon', (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                elif corners == 6:
                    cv2.putText(imgContour, 'Hexagon', (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                elif corners >= 7:
                    cv2.putText(imgContour, 'Circle', (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.imshow('img', img)
        cv2.imshow('imgcontour', imgContour)
        # cv2.waitKey(10000)
    else:
        break
    if cv2.waitKey(500) == ord('q'):
        break