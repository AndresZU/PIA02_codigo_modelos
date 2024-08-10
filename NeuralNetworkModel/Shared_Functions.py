# import OpenCV for webcam capture and mediapipe for landmark detection
import cv2
import mediapipe as mp

# Import numpy to create data arrays, os to creata data directories
import numpy as np
import os

# Define mediapipe capture model and initialize drawing to show hand and arm landmarks
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Path for exported numpy arrays with the keypoints
DATA_DIR = os.path.join('Data') 

# LESCO signs we want to detect
signs = np.array(['Letra A', 'Letra B', 'Letra J'])

# Capture 3 videos per sign
no_sequences = 15

# Each video is 30 frames (images) per lenght
sequence_length = 15

# Function to process image and capture landmarks
def mediapipe_detection(image, model):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Lock image to prevent changes
    image.flags.writeable = False
    # Process image with model                
    results = model.process(image)
    # Unlock image                 
    image.flags.writeable = True
    # Convert back to BGR                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to style the landmarks detected
def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# Function to extract keypoint data from the results
def extract_keypoints(results):
    # Extract pose keypoints using a python list comprehension, if no pose landmark is detected, we will create a numpy array with 33*4 zeros
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Extract face keypoints using a python list comprehension, if no pose landmark is detected, we will create a numpy array with 468*3 zeros
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # Extract left hand keypoints using a python list comprehension, if no pose landmark is detected, we will create a numpy array with 21*3 zeros
    leftHand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Extract right hand keypoints using a python list comprehension, if no pose landmark is detected, we will create a numpy array with 21*3 zeros
    rightHand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, leftHand, rightHand])