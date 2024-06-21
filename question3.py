import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

# Define the model
model = Sequential()

# Add the first convolutional layer
model.add(Conv2D(2, kernel_size=(2, 2), activation='relu', padding='same', strides=2, input_shape=(5, 5, 1)))

# Add the max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add the second convolutional layer
model.add(Conv2D(2, kernel_size=(2, 2), activation='relu', padding='same', strides=2, input_shape=(3, 3, 2)))

# Add the average pooling layer
model.add(AveragePooling2D(pool_size=(2, 2)))

# Print the model summary
model.summary()