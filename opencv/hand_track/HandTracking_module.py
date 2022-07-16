import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, model_complexity = 0,
                detectionCon = 0.5, trackCon = 0.5):

        # self.mode = mode
        # self.maxHands = maxHands
        # self.model_complexity = model_complexity,
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHands, model_complexity, detectionCon, trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame   

    def findPosition(self, frame):
        lmList = []
        if self.results.multi_hand_landmarks:
            # myHand = self.results.multi_hand_landmarks[handNo]
            # ==============Classify Left or Right hand====================
            # if self.results.multi_handedness:
            #     for hand_handedness in self.results.multi_handedness:                    
            #         whichHand = hand_handedness.classification[0].label
            #         if whichHand == "Left":
            #             print('Right')
            #         else:
            #             print('Left')                    

            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if id==4:
                        cv2.circle(frame, (cx,cy),7, (255,0,255), cv2.FILLED)
        return  lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()