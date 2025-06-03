import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the dataset
try:
    df = pd.read_csv("Deep_learning_01/insurance_data.csv")
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

print("First 8 rows of data:")
print(df.head(8))

# Split the data
x = df[["age", "affordability"]]
y = df["buy"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Build and train the TensorFlow model
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation="sigmoid", kernel_initializer="ones", bias_initializer="zeros")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=500)

# Evaluate the model
print("\nEvaluation on test data:")
print(model.evaluate(x_test, y_test))

# Extract weights and biases
weights, biases = model.layers[0].get_weights()
print("Weights:", weights)
print("Biases:", biases)

# Predict using custom sigmoid model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(age, affordability, coef, intercept):
    weighted_sum = coef[0] * age + coef[1] * affordability + intercept
    return sigmoid(weighted_sum)

# Log loss implementation
def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))

# Sigmoid for vector input
def sigmoid_numpy(X):
    return 1 / (1 + np.exp(-X))

# Manual gradient descent implementation
def gradient_descent(age, affordability, y_true, epochs, loss_threshold):
    w1 = w2 = 1.0
    bias = 0.0
    learning_rate = 0.1
    n = len(age)

    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordability + bias
        y_predicted = sigmoid_numpy(weighted_sum)
        loss = log_loss(y_true, y_predicted)

        # Gradients
        w1_grad = (1 / n) * np.dot(age.T, (y_predicted - y_true))
        w2_grad = (1 / n) * np.dot(affordability.T, (y_predicted - y_true))
        bias_grad = np.mean(y_predicted - y_true)

        # Update weights
        w1 -= learning_rate * w1_grad
        w2 -= learning_rate * w2_grad
        bias -= learning_rate * bias_grad

        print(f"Epoch {i}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {bias:.4f}, loss = {loss:.4f}")

        if loss <= loss_threshold:
            break

    return w1, w2, bias

# Prepare data for gradient descent
age = x_train[:, 0]
affordability = x_train[:, 1]
y_train_array = y_train.values

# Run manual gradient descent
w1, w2, bias = gradient_descent(age, affordability, y_train_array, epochs=500, loss_threshold=0.1582)
