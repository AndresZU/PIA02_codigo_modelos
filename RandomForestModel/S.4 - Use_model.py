# Import pickle to read the model file and save to local variable
import pickle

# Import Cv2 for webcam capture, Mediapipe for landmark detection and numpy to easily handle the model prediction.
import cv2
import mediapipe as mp
import numpy as np

# Reading model file into local variable
model_dict = pickle.load(open('./ml_model.randomForest', 'rb'))
model = model_dict['model']

# Defining webcam object to initialize.
cap = cv2.VideoCapture(0)

# Define hand object from Mediapipe to capture hand landmarks
mp_hands = mp.solutions.hands

# Define drawing utils and styles object to create rectable around the detected hand.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Creating a dictionary to link the LESCO sign to the index used to train the model.
labels_dict = {0: 'Letra A', 1: 'Letra B', 2: 'Letra C'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    Height, Width, _ = frame.shape

    # Convert frame captured to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark from the captured frame
    results = hands.process(frame_rgb)

    # If results variable has values, meaning, at least 1 hand is detected.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmark points on the detected hand.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Create landmark array and normalize with the same process used as before.
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Define size for box outside the hand.
        x1 = int(min(x_) * Width) - 10
        y1 = int(min(y_) * Height) - 10

        x2 = int(max(x_) * Width) - 10
        y2 = int(max(y_) * Height) - 10

        # Use the model for inference and save results on variable.
        prediction = model.predict([np.asarray(data_aux)])

        # Get the specific sign from the previously defined dictionary based on the model output.
        predicted_sign = labels_dict[int(prediction[0])]

        # Draw box arround the hand.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
