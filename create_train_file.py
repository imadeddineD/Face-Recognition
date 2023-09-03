import os 
import numpy as np 
import cv2 as cv 

people =  ['Ben Afflek' ,  'Maddona' , 'Mindy Kaling'] # list of names of people whose faces will be trained on

our_Path = r'C:\Users\LENOVO\Desktop\p_face-recognition\train' # the path to the folder where the training images are stored.

haar_cascade = cv.CascadeClassifier("harr_cascade_face.xml") # pre-trained classifier that is used to detect faces in an image.

features = [] # list that will contain image arrays of faces .
labels = [] # list that will contain corresponding labels for each face .

def creat_train() : 
    for person in people : 
        path = os.path.join(our_Path , person) # path to the folder containing the images of the current person.
        label = people.index(person) # the label assigned to the current person.
        
        for img in os.listdir(path) : # returns a list of the names of all files and directories in the directory given by 'path'.
            img_path = os.path.join(path , img)
            
            img_array = cv.imread(img_path) # reads an image from the specified file path and returns a NumPy array of pixel values.
            grayImg = cv.cvtColor(img_array , cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(grayImg , scaleFactor = 1.1 , minNeighbors = 1) #detects the face in the grayscale image.
            
            for (a,b,c,d) in faces_rect : # loop loops through each detected face and extracts the region of interest (ROI) from the grayscale image.
                faces_roi = grayImg[b:b+d , a:a+c]
                features.append(faces_roi) # The ROI is appended to the features list and the corresponding label is appended to the labels list.
                labels.append(label)
                
creat_train()

# converts the features and labels lists to NumPy arrays : 
features = np.array(features , dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create() # creates an instance of a face recognition algorithm.

face_recognizer.train(features , labels) # trains the algorithm on the features and labels.

face_recognizer.save('face_trained.yml') # saves the trained model to a file named face_trained.yml.