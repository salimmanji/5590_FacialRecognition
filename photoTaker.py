import cv2

# User input for starting image numbering.
imgNum = int(input("Enter starting image number: "))

print("Camera Starting.")
camera = cv2.VideoCapture(0)
cv2.namedWindow("Image")

# Basic loop. While true, keep taking photos using OpenCV.
wantContinue = True
while (wantContinue):
    ret, frame = camera.read()
    if not ret:
        print("Failed to get frame. Exiting...")
        break

    cv2.imshow("Image", frame)
    fileName = "/home/pi/COMP5590/test-data/" + "test" + str(imgNum) + ".jpg"
    cv2.imwrite(fileName, frame)
    print(fileName +  "written successfully.")
    imgNum+=1
    print("Press any key to take more images, or press 'q' to quit.")
    
    k = cv2.waitKey(0)
    if k == ord('q'):
        wantContinue = False
        break

# Close camera and OpenCV assets.
camera.release()
cv2.destroyAllWindows()
print("Goodbye.")