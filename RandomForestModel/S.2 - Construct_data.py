# Import os to interact with local directories, pickle to create data files.
import os
import pickle

# Import cv2 to read the images captures in the previous step.
# Import mediapipe to get the landmark data out of the images.
import cv2
import mediapipe as mp

# Mediapipe class to get hands information
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Reference to image folder
IMG_DIR = './images'

# Create to arrays, one for the landmark data of the images, and one for the labels (The sign each image represents)
data = []
labels = []

# Loop all different sub directories in the images folder.
for dir_ in os.listdir(IMG_DIR):
    for img_path in os.listdir(os.path.join(IMG_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []
        
        # Read the images of the sub directory.
        img = cv2.imread(os.path.join(IMG_DIR, dir_, img_path))
        # Convert image to RGB since mediapipe requires it.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Analyze image using the mediapipe class.
        mp_results = hands.process(img_rgb)

        # Confirm if image contains hands.
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                # Getting hand landmark values on the X and Y axis.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                
                # subtracting the minimum point from the X and Y values to normalize data.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save dictionary of arrays into a pickle serialized file.
f = open('dataset.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(len(data))
print(len(labels))
