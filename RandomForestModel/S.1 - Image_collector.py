# Import os for interacting with local directories and cv2 for image capture.
import os
import cv2

# Define a local directory for the images to land, and create it if it doesn't exist already
IMG_DIR = './images'
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Define the number of signs to capture, and the amount of pictures for each one.
# This will create sub directories for each sign in the images folder, starting from 0.
# We will define which sign relates to which index later.
num_signs = 11
sign_size = 100

# Define a new video capture source, webcam 0
cap = cv2.VideoCapture(0)

# Loop over the number of signs to capture
for i in range(num_signs):
    # Create the sub directory for each sign
    if not os.path.exists(os.path.join(IMG_DIR, str(i))):
        os.makedirs(os.path.join(IMG_DIR, str(i)))

    # Information for user
    print('Collecting images for sign {}'.format(i))

    done = False
    # Wait for user confirmation to capture, this will allow the user to prepare between signs
    # In this case we prompt the user to press the letter S to start, but this can be modified to any other keyboard letter.
    # Things like the fond type, size, color and position can also be adjusted.
    # Using neon green for text color and avoiding using black or white which can be confused with the background.
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready to capture, press S to start', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break
    
    # Define counter variable to give all images a unique name
    capture_counter = 0
    # Capture a new image every 25 miliseconds and save to sub directory.
    while capture_counter < sign_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Write to sub directory.
        cv2.imwrite(os.path.join(IMG_DIR, str(i), '{}.jpg'.format(capture_counter)), frame)

        capture_counter += 1

# Kill OpenCV window and release the webcam.
cap.release()
cv2.destroyAllWindows()
