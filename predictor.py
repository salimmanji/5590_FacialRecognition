import cv2
import os
import numpy as np

# Function to detect faces using OpenCV. The test image is converted to greyscale, and the 'Haarcascade frontal face' file is opened. This file
# contains the coordinates to map an individual's face which are compared against the test image. When a face has been found, the face area is extracted
# and returned. If no face was found, the original image is returned.
#
# @param img is the image to test.
# @return is the facial portion of the image if a face was detected, else the original image.
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image = gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]


# This function draws a rectangle over the original image, outlining a face.
#
# @param img is the image onto which to draw the rectangle.
# @param rect are the coordinates of the rectangle encapsulating a face.
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
# This function adds text to the image, just above the rectangle drawn above. 
#
# @param img is the image onto which to draw the desired text.
# @param text will be mounted onto the image
# @param x the x-coordinate for the text to start.
# @param y the y-coordinate for the text to start.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# This function recognizes the user trained in the each of the models. If a face is recognized, a rectangle will
# be drawn around the face and the label is mounted before being saved and returned. If no image is detected, 
# a copy is saved in failures folder for later reference.
#
# @param test_img is the image to test against the facial recognition model.
# @param face_recognizer is the recognizer to use.
# @param face_rec_name is the name of the recognizer for file naming purposes.
# @param file_name is used in file naming.
# @return img is the test image with either a rectangle and text overlayed, or the original.
def predict(test_img, face_recognizer, face_rec_name, file_name):
    img = test_img.copy()
    face, rect = detect_face(img)

    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label] + " - Score: " + str(confidence)
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
        cv2.imwrite("/home/pi/COMP5590/recognized-images/" + face_rec_name + "/" + file_name + "-" + label_text + '_' + ".jpg", img)
    else:
        print("No face(s) found.")
        cv2.imwrite("/home/pi/COMP5590/recognized-images/"  + face_rec_name + "/failures/" + file_name, img)

    return img

# Function to show image with prediction values on the screen.
# 
# @predictedImage is the image to resize and briefly show.
def show_image(predictedImage):
    cv2.imshow(subjects[1], cv2.resize(predictedImage, (400, 500)))
    cv2.waitKey(100)
    cv2.destroyAllWindows()


subjects = ["", "Salim "]

# Open and read face recognizers built in facialRecognition.py script.
LBPHF_recognizer = cv2.face.LBPHFaceRecognizer_create()
LBPHF_recognizer.read("/home/pi/COMP5590/model/LBPHF.yml")
# eigenFace_recognizer = cv2.face.EigenFaceRecognizer_create()
# eigenFace_recognizer.read("/home/pi/COMP5590/model/EigenFace.yml")
# fisherFace_recognizer = cv2.face.FisherFaceRecognizer_create()
# fisherFace_recognizer.read("/home/pi/COMP5590/model/FisherFace.yml")
print("Predicting images...")

subject_dir_path = "/home/pi/COMP5590/test-data/"
subject_images_names = os.listdir(subject_dir_path)

# For each loop to open every image in the test-data folder, read the image and try and recognize a face.
# Hidden files (starting with a ".") are ignored.
for image_name in subject_images_names:
    
    if image_name.startswith("."):
        continue
    
    image_path = subject_dir_path + "/" + image_name
    test_img = cv2.imread(image_path)

    predicted_img_LBPHF = predict(test_img, LBPHF_recognizer, "LBPHF", image_name)
    show_image(predicted_img_LBPHF)
    # predicted_img_Eigen = predict(test_img, eigenFace_recognizer, "EigenFace", image_name)
    # show_image(predicted_img_Eigen)
    # predicted_img_Fisher = predict(test_img, fisherFace_recognizer, "FisherFace", image_name)
    # show_image(predicted_img_Fisher)

    print(image_name + " prediction complete...")

print("All predictions complete!")
