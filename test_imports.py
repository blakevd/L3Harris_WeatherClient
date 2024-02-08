## This is terminal code for learning about ML ##


import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate dummy data
# Assuming you have a sequence of integers every hour
# You can replace this with your actual dataset
# Here, we generate 100 sequences, each containing integers from 0 to 9
num_sequences = 100
sequence_length = 10
num_classes = 10  # integers from 0 to 9
sequences = np.random.randint(0, num_classes, size=(num_sequences, sequence_length))
next_integers = sequences[:, -1]  # Next integer is the last element of each sequence

# Reshape the sequences to fit the LSTM input shape
sequences = sequences.reshape(-1, sequence_length, 1)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 1)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(sequences, next_integers, epochs=10, batch_size=32, validation_split=0.2)

# Generate a second set of dummy data
num_sequences_2 = 100
sequences_2 = np.random.randint(0, num_classes, size=(num_sequences_2, sequence_length))
next_integers_2 = sequences_2[:, -1]

# Reshape the sequences to fit the LSTM input shape
sequences_2 = sequences_2.reshape(-1, sequence_length, 1) #infers size of first dimension auto based on number of elements

# Train the model on the second set of data
model.fit(sequences_2, next_integers_2, epochs=10, batch_size=32, validation_split=0.2)

# Once trained, you can use the model to predict the next integer in a sequence
# Example of a new sequence
new_sequence = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])  # Replace this with your actual new sequence

# Ensure the shape of the new sequence matches the input shape expected by the model
new_sequence = new_sequence.reshape(1, sequence_length, 1)  # Reshape to (1, sequence_length, 1)

# Make predictions
predicted_probabilities = model.predict(new_sequence)  # Predicted probabilities for each class
predicted_integer = np.argmax(predicted_probabilities)  # Predicted integer based on highest probability

print("Predicted integer:", predicted_integer)