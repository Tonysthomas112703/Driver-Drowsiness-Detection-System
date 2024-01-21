import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Eye aspect ratio (EAR) calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define dlib's face detector, facial landmark predictor, and set the threshold for drowsiness
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
drowsy_threshold = 0.3

# Open the Pi camera for video capture
camera = cv2.VideoCapture(0)

# Loop through each frame of the video
while True:
    # Read the video frame and convert to grayscale
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_detector(gray, 0)
    
    # Loop through each face detected
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        # Extract the left and right eye landmarks and calculate the eye aspect ratio (EAR)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average the EAR of both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if the EAR falls below the drowsy threshold
        if avg_ear < drowsy_threshold:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw the facial landmarks and the computed EAR on the frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(frame, "EAR: {:.2f}".format(avg_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show the video frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
camera.release()
cv2.destroyAllWindows()
