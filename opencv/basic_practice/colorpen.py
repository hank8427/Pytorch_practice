import cv2
import numpy as np

penColor = [[113,128,0,143,184,101],
            [38,78,0,104,255,93],
            [0,179,37,4,255,255]]

drawColor = [[255,0,0],
             [0,255,0],
             [0,0,255]]

# [x,y,color]
draw_position=[]

def pen(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(len(penColor)):
        lower = np.array(penColor[i][:3])
        upper = np.array(penColor[i][3:6])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_or(img, img, mask = mask)
        penx, peny = contours(mask)
        cv2.circle(imgContour, (penx,peny), 10, drawColor[i], cv2.FILLED)
        if peny != -1:
            draw_position.append([penx, peny, i])
        # cv2.imshow('result', result)

def contours(img):
    contours, herarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = -1, -1, -1, -1
    for cnt in contours:
        # cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
        area = cv2.contourArea(cnt)
        if area > 120:
            peri = cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)
            x,y,w,h = cv2.boundingRect(vertices)
    return x+w//2, y

def drawpen(draw_position):
    for point in draw_position:
        cv2.circle(imgContour, (point[0],point[1]), 10, drawColor[point[2]], cv2.FILLED)

vc = cv2.VideoCapture(0)
while True:
    ret, frame = vc.read()
    if ret:
        imgContour = frame.copy()
        pen(frame)
        drawpen(draw_position)
        # cv2.imshow("video",frame)
        cv2.imshow("contour",imgContour)
    else:
        break
    if cv2.waitKey(50) == ord('q'):
        break