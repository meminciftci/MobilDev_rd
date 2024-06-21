import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('image.jpg')                # Loading the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the image to grayscale

# Detecting faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Drawing rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
# Displaying the image with detected faces
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()