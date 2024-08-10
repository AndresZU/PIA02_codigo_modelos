# import OpenCV for webcam capture and mediapipe for landmark detection
import cv2
import mediapipe as mp

# Import numpy to create data arrays, os to creata data directories
import numpy as np
import os

# Import functions from separate file to process the data
from Shared_Functions import mediapipe_detection, draw_styled_landmarks, extract_keypoints, signs, no_sequences, sequence_length, DATA_DIR

# Define mediapipe capture model and initialize drawing to show hand and arm landmarks
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create directoy and sub folders
for action in signs: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_DIR, action, str(sequence)))
        except:
            pass


# Collect videos for each sign
# Define webcam to use
cap = cv2.VideoCapture(0)
# Configure mediapipe model parameters
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through signs
    for sign in signs:
        # Loop through each video (30)
        for sequence in range(no_sequences):
            # Loop through video frames (images, 30)
            for frame_num in range(sequence_length):
                # Read camera feed
                ret, frame = cap.read()
                # Gather keypoint and landmark data using mediapipe
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks on the screen
                draw_styled_landmarks(image, results)
                
                # Give user a warning message when recording of a new video is about to start
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING RECORDING', (120,200), 
                               cv2.FONT_HERSHEY_DUPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting images for {} Video Number {}'.format(sign, sequence), (15,12), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign, sequence), (15,12), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Save keypoints from videos into a local variable
                keypoints = extract_keypoints(results)
                # Set path for files to land
                npy_path = os.path.join(DATA_DIR, sign, str(sequence), str(frame_num))
                # Save detected keypoints in numpy format into file on the specific path
                np.save(npy_path, keypoints)

                # Stop process if the key 's' is pressed
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    break
        # Prompt user to press c to continue capturing next sign
        cv2.putText(image, 'Capture finished, press "c" to continue', (120,200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        cv2.waitKey(0)
    
    # Stop capture process and kill webcam capture
    cap.release()
    cv2.destroyAllWindows()