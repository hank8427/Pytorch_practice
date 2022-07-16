import cv2

# img = cv2.imread('d:/PY_file/opencv/man_face.jpg')
vc = cv2.VideoCapture(0)
while True:
    ret, img = vc.read()
    if ret:
        img = cv2.resize(img, (0,0), fx=1, fy=1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_model = cv2.CascadeClassifier('d:/PY_file/opencv/face_model.xml')
                            #                   縮小倍率, 被幾個框偵測到
        faceRect = face_model.detectMultiScale(gray, 1.1, 5)
        print(faceRect)
        for(x, y ,w, h) in faceRect:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),3)

        cv2.imshow('face',img)
    else:
        break
    if cv2.waitKey(50) == ord('q'):
        break