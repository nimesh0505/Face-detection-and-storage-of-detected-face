import cv2
import sys


webcam = cv2.VideoCapture(0) #Use camera 0
face = cv2.CascadeClassifier('train_face.xml')

while True:
    ret,frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(frame,1.3,5)
    padding = 10

    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x-padding, y-padding), (x+w+padding, y+h+padding), (255, 0, 0), 2)
        sub_face = frame[y:y+h+padding, x:x+w+padding]
        #sub_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
        FaceFileName = "facesCapture/face_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName,sub_face)
        cv2.imshow("Detecting and storing face",frame)
    if cv2.waitKey(1)== ord('q'):
            break
webcam.release()
cv2.destroyAllWindows()



           
