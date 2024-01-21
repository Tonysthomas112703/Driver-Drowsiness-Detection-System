# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils, resize
import imutils
import dlib
import cv2
import playsound
import subprocess
import RPi.GPIO as GPIO

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to shut down the Raspberry Pi
def shutdown():
    subprocess.call("sudo shutdown now", shell=True)

# Constants and initialization
thresh = 0.25
frame_check = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # Update the path to your shape_predictor.dat file

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)

flag = 0

# GPIO setup for a switch connected to GPIO 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Main loop for capturing and processing video frames
while True:
    # Check if the switch is pressed to trigger shutdown
    if GPIO.input(18) == GPIO.LOW:
        shutdown()
        break

    # Read a frame from the video capture
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)  # Resize the frame for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)  # Detect faces in the grayscale image

    # Loop through detected faces and process eyes
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # Convert face landmarks to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                playsound.playsound("wake_up_sound.mp3")  # Play a sound to wake up the user
        else:
            flag = 0  # Reset the counter if eyes are open

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to exit the program
        break

# Clean up resources and close windows
cv2.destroyAllWindows()
cap.release()
