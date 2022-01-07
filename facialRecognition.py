import cv2
import os
import numpy as np
from pathlib import Path

# Function to detect face using OpenCV. Image is converted to gray-scale, then a face detector is run. If a face is found, the face is returned which can be used in recognizing faces.
# 
# @param img is the image from which to find faces.
# @return If a face was found, return it, else return the original image. 

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image = gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]


# This function create two arrays, one to hold the facial data, the other to store respective subject lables. 
# Since the training data folders start with the letter 's', any folders not beginning with an 's' are not included. Once a folder
# is found, remove the preceeding char. With hidden files avoided (names starting with a '.'), each image is loaded, and facial detection is handled. 
# If a face is detected, add the label and the facial data to the respective arrays.  
#
# @param data_folder_path is the directory where the training image files are stored.
# @return faces is the array of facial data of the subject.
# @return labels is the labels array associated with the subject's images.
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:       
            if image_name.startswith("."):
                continue
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (400, 500))
            cv2.imshow("Training on image...", resized_image)
            cv2.imwrite(image_path, resized_image)
            cv2.waitKey(100)
            face, rect = detect_face(resized_image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

# Function to initiate model training and save the file. To fully test the capabilities of the system, I am training three recognizers:
# Local Binary Patterns Histogram (LBPH), EigenFaces and FisherFaces. Since the file likely does not exist, some simple OS commands are used
# to touch the file, then save.
#
# @param recognizerName is the name of the recognizer for user feedback and file naming.
# @param recognizer is the facial recognizer to use.
def trainData(recognizerName, recognizer):
    recognizer.train(faces, np.array(labels))
    Path("/home/pi/COMP5590/model/" + recognizerName + ".yml").touch()
    recognizer.save("/home/pi/COMP5590/model/" + recognizerName + ".yml")
    print("New " + recognizerName + ".yml file saved.")


subjects = ["", "Salim "]

print("Preparing data...")
faces, labels = prepare_training_data("/home/pi/COMP5590/training-data/")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

LBPHF_recognizer = cv2.face.LBPHFaceRecognizer_create()
trainData("LBPHF", LBPHF_recognizer)
#eigenFace_recognizer = cv2.face.EigenFaceRecognizer_create()
#trainData("EigenFace", eigenFace_recognizer)
#fisherFace_recognizer = cv2.face.FisherFaceRecognizer_create()
#trainData("FisherFace", fisherFace_recognizer)

print("Model(s) built. Please run predictor.py to predict!")
