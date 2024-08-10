# Import pickle to open file with data.
import pickle

# Import sklearn libraries to create the train_test split, train the model, and evaluate the accuracy.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import numpy to convert from original array format to numpy format.
import numpy as np

# Read data file.
data_file = pickle.load(open('./dataset.pickle', 'rb'))

# Separate data from labels.
data = np.asarray(data_file['data'])
labels = np.asarray(data_file['labels'])

# Create train-test split.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the model with default hyperparameters
model = RandomForestClassifier()

# Train the model.
model.fit(x_train, y_train)

# Use the trained model to predict real data.
y_predict = model.predict(x_test)

# Calculate accuracy score of the model based on prediction.
score = accuracy_score(y_predict, y_test)

# Print score to console.
print('{}% of samples were classified correctly !'.format(score * 100))

# Save model into file using pickle.
f = open('ml_model.randomForest', 'wb')
pickle.dump({'model': model}, f)
f.close()
