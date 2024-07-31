import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten # type: ignore
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the data
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')
X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

# Reshape and normalize the data
X_train = X_train.reshape(len(X_train), 100, 100, 3) / 255.0
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3) / 255.0
Y_test = Y_test.reshape(len(Y_test), 1)

print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)

# Display a random training image
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()

# Train a CNN model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(X_train, Y_train, epochs=5, batch_size=64)
cnn_model.evaluate(X_test, Y_test)

# Prepare data for SVM
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train an SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_flat, Y_train.ravel())

# Evaluate the SVM model
y_pred_svm = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(Y_test, y_pred_svm)
print("SVM Accuracy: ", svm_accuracy)

# Display a random test image
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

# Predict with the SVM model
y_pred_svm_single = svm_model.predict(X_test_flat[idx2].reshape(1, -1))
y_pred_svm_proba = svm_model.predict_proba(X_test_flat[idx2].reshape(1, -1))[0][1] > 0.5

if y_pred_svm_single == 0:
    pred = 'dog'
else:
    pred = 'cat'
    
print("Our model says it is a :", pred)