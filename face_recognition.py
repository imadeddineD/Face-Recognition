import cv2 as cv 
import numpy as np 

haar_cascade = cv.CascadeClassifier("harr_cascade_face.xml")

people = ['Ben Afflek' ,'Maddona' , 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(r'C:\Users\LENOVO\Desktop\p_face-recognition\val\Random\ran1.jpg')

grayImg = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

# " detect the face in the image " 
faces_rect = haar_cascade.detectMultiScale(grayImg , scaleFactor = 1.1 , minNeighbors = 1)

for (a,b,c,d) in faces_rect : # loop loops through each detected face and extracts the region of interest (ROI) from the grayscale image.
                faces_roi = grayImg[b:b+d , a:a+c]
                
                label , confidence = face_recognizer.predict(faces_roi)
                
                cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
                cv.rectangle(img, (a,b), (a+c,b+d), (0,255,0), thickness=2)

cv.imshow("Detected image" , img)

cv.waitKey(0)