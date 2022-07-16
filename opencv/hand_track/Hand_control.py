import cv2
import time
import HandTracking_module
import math
import pyautogui

w, h = pyautogui.size()
wCam, hCam = w//2, h//2

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = HandTracking_module.handDetector()
while True:
    mx, my = pyautogui.position()
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
        # print(lmList[4])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), thickness=3)
        cv2.circle(frame, (x1,y1), 12, (255,0,255), cv2.FILLED)
        cv2.circle(frame, (x2,y2), 12, (255,0,255), cv2.FILLED)
        cv2.circle(frame, (cx,cy), 12, (255,0,255), cv2.FILLED)
        
        pyautogui.moveTo(abs(w-cx*2), cy*2)

        length = math.hypot(x1-x2,y1-y2)
        if length <=30:
            cv2.circle(frame, (cx,cy), 12, (255,255,0), cv2.FILLED)
            pyautogui.click(button='left')
            # pyautogui.click(clicks = 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(frame, f'FPS:{int(fps)}',(10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 3)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)