import cv2

# Load the video
video_path = "../video.mp4"             # Video path on Windows
cap = cv2.VideoCapture(video_path)      # Loading the video

# Creating a face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Reading the video frame by frame
while True:
    ret, frame = cap.read()     # Read the frame
    
    if not ret:                 # If no frame is available, break
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the frame to grayscale

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detecting faces in the frame

    for (x, y, w, h) in faces:  # Drawing rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)  # Displaying the frame with detected faces

cap.release()               # Releasing the video capture
cv2.destroyAllWindows()     # Closing the window