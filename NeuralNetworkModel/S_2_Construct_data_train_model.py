# Import scikit-learn library to perform train-test split and tensorflow.keras to convert data to binary
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import model training library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Import libraries to evaluate model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import numpy as np
import os

# Import variables from previous step
from Shared_Functions import signs, no_sequences, sequence_length, DATA_DIR

# Create a label map for the signs in the variable
label_map = {label:num for num, label in enumerate(signs)}

# Load numpy files to program and process into data and labels
data, labels = [], []
for sign in signs:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_DIR, sign, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        data.append(window)
        labels.append(label_map[sign])

# Save to variable
X = np.array(data)

# Save to variable
y = to_categorical(labels).astype(int)

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set local directory to save logs
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Set layers of the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

# Configure model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Get model summary
model.summary()

# Save model to file
model.save('model.keras')

# Evaluate model

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# print(multilabel_confusion_matrix(ytrue, yhat))

score = accuracy_score(ytrue, yhat)
# Print score to console.
print('{}% of samples were classified correctly !'.format(score * 100))