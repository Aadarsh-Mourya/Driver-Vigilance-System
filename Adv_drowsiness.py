from scipy.spatial import distance
import tensorflow as tf
from imutils import face_utils
import imutils
import dlib
import cv2

# Initialize the mixer for playing sounds
from pygame import mixer
mixer.init()
mixer.music.load("music.wav")

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

# Function to calculate the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3 * D)
    return mar

# Threshold values and frame checks
ear_threshold = 0.25
mar_threshold = 0.55
frame_check = 20
yawn_frame_check = 15
head_bow_check = 15

# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get indexes for the left and right eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]
(cStart, cEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]

# Initialize the webcam video capture
cap = cv2.VideoCapture(0)
flag = 0
yawn_flag = 0
head_bow_flag = 0

# Calibrate the head orientation using the initial frames
calibration_frames = 30
initial_vertical_difference = 0
count = 0

while count < calibration_frames:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the vertical difference between the nose and chin for calibration
        nose_point = shape[nStart:nEnd][4]
        chin_point = shape[cStart:cEnd][8]
        vertical_difference = abs(nose_point[1] - chin_point[1])
        
        initial_vertical_difference += vertical_difference
        count += 1
    
# Calculate the average initial vertical difference
initial_vertical_difference /= calibration_frames

# Set the threshold based on the initial vertical difference (e.g., 20% higher or lower than the initial difference)
head_bow_threshold = initial_vertical_difference * 1.20

# Start video processing
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate eye aspect ratios
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2
        
        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check for drowsiness based on the eye aspect ratio
        if ear < ear_threshold:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
        
        # Calculate mouth aspect ratio (MAR)
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        # Draw mouth contours
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Check for yawning based on the mouth aspect ratio
        if mar > mar_threshold:
            yawn_flag += 1
            print(yawn_flag)
            if yawn_flag >= yawn_frame_check:
                cv2.putText(frame, "*************YAWN ALERT!*************", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            yawn_flag = 0
        
        # Calculate the vertical difference between the nose and chin
        nose_point = shape[nStart:nEnd][4]  # The nose tip point
        chin_point = shape[cStart:cEnd][8]  # The chin tip point
        vertical_difference = abs(nose_point[1] - chin_point[1])
        
        # Check for head bowing based on the vertical difference
        if vertical_difference > head_bow_threshold:
            head_bow_flag += 1
            print(head_bow_flag)
            if head_bow_flag >= head_bow_check:
                cv2.putText(frame, "************HEAD BOW ALERT!************", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            head_bow_flag = 0
    
    # Display the processed frame
    cv2.imshow("Frame", frame)
    
    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up resources and close windows
cv2.destroyAllWindows()
cap.release()
