import cv2
import time
import HandTracking_module

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = HandTracking_module.handDetector()
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)