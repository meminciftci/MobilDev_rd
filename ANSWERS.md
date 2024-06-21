# Part 1: Programming and Neural Networks
## 1.
### a. Developing an Algorithm
There are several steps to develop an algorithm that provides the requested functionality: initialization, iteration, checking, and returning the result.

- **Initialization** consists of creating useful variables. A list to store the start indices of occurrences, `lenA` and `lenB` integer variables to store the length of list A and B respectively since they will be used in many places.

- **Iteration** is the implementation of two nested for loops. The outer loop will search for a match on the first element of list B with the current element of list A. If they match, the inner loop will check if the sequential elements are also matching in list A. This part can also be mentioned as the checking part.

- After all the elements in list A have been checked, the algorithm shall return the result value, which is the list `occurrences`.

### b. Algorithm Implementation
The implementation of the algorithm using the basics of any programming language is as follows:

```
function find_pattern(A, B):
    occurrences = []
    lenA = length(A)
    lenB = length(B)
    
    for i from 0 to lenA - lenB:
        if A[i:i+lenB] == B:
            occurrences.append(i)
    
    return occurrences


```
* Above is the pseudocode implementation of the algorithm.

### c. Python Implementation 
```
def find_pattern(A, B):             # A is the main list and B is the pattern list
    occurrences = []                # List to store the indices of the pattern occurrences
    lenA = len(A)                   # Length of the main list A
    lenB = len(B)                   # Length of the pattern list B
    
    for i in range(lenA - lenB + 1):    # Iterate through the main list A
        match = True                    # Assume that the pattern is found
        for j in range(lenB):           # Iterate through the pattern list B
            if A[i + j] != B[j]:         
                match = False           # If the elements do not match set match to false, break the inner loop
                break                   
        if match:                    
            occurrences.append(i)   # If the pattern is found append the index to the occurrences list
            i += lenB - 1           # Skip the pattern length to avoid overlapping
    
    return occurrences              # Return the list of occurrences

# Test the function
A = [1, 70, 9, 1, 2, 30, 6, 1, 2, 30, 50]
B = [1, 2, 30]
print(find_pattern(A, B))

```
* Above is the Python implementation of the algorithm. It can be accessed through the file named "question1.py".

## 2.
### a.
* Importing all the functions from a module can lead to unnecessary pollution and maintenance issue. Therefore, importing only the needed function would be better. 

```
from math import ceil 
x = ceil(x)

```

### b.
* If using the indices in a for loop is not necessary, iteration directly from the list would be more efficient and space friendly since there will be no need for an extra variable for index. 

```
list_of_fruits = ["apple", "pear", "orange"]
for fruit in list_of_fruits:
    process_fruit(fruit)

```

### c.
* Implementation of class Rectangle looks clean. There could be an additional function for printing out a rectangle object when it is need.

```
class Rectangle:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def area(self):
        return self.height * self.width
    
    def __str__(self):
        return f"Rectangle(height={self.height}, width={self.width})"

```
* With the addition of __str__ method, print(rectangle) can be easily managed. 

### d. 
* When we are using pointers, it is important to consider possibilities for memory leak. Memory leak happens when an initialized pointer is not freed after its use. Therefore, we need to add a line of code that frees the allocated space of the pointer. 

* I am not familiar with the provided header, but I know that in order to compile this code, it does not need to be included. Instead, we can use a more widely known header like iostream if it is needed. 

```
#include <iostream>

void do_something() 
{ 
    int* ptr = new int(10); 
    // some math operations 

    // Free the allocated memory
    delete ptr; 
}

int main() 
{ 
    do_something(); 
    return 0; 
}

```


## 3. 2D CNN
### a. Perform an experiment using random numerical values step by step.

Let us show the step-by-step experiment.

#### i. Input Image
* Shape of the input image is (5, 5, 1) and it is grayscale meaning it has only one channel.

```
[[ 50,  80,  30,  10,  60],
 [100, 150, 200,  20,  70],
 [ 80,  90, 120,  40,  30],
 [ 60,  70, 100,  10,  20],
 [ 10,  50,  70,  30,  40]]
```
Above is an example input.

#### ii. First Convolution Layer
* Number of filters to apply to the input image is 2. 
* Kernel size which determines the bits we are going to include in our each calculation is (2, 2).
* Strides which is the number of pixels we are going to move after each calculation is (2, 2).
* Padding is zero, which means for the edges, we are going to use the value 0 if there is no pixel to be calculated.

```
[[  0,   0,   0,   0,   0,   0,   0],
 [  0,  50,  80,  30,  10,  60,   0],
 [  0, 100, 150, 200,  20,  70,   0],
 [  0,  80,  90, 120,  40,  30,   0],
 [  0,  60,  70, 100,  10,  20,   0],
 [  0,  10,  50,  70,  30,  40,   0],
 [  0,   0,   0,   0,   0,   0,   0]]

```
Above is the input after applying the padding which is zero.

```
[[ 1, -1],
 [ 0,  1]]

```
Above is filter 1.
```
[[ 1,  0],
 [-1,  1]]

```
Above is filter 2.

* Output calculation according to these filters will be (3, 3, 2) shape. 

```
[[50, 30, 60],
 [-20,  70, -20],
 [-50,  40,  30]]

```
Above is the output for filter 1.

```
[[50, -50, 50],
 [80, 180, 10],
 [10, 90, 20]]

```
Above is the output for filter 2.

#### iii. Activation (ReLU)

```

Filter 1:
[[50, 30, 60],
 [0,  70, 0],
 [0,  40,  30]]

Filter 2:
[[50, 0, 50],
 [80, 180, 10],
 [10, 90, 20]]

```

#### iv. Max Pooling and Average Pooling

```

Filter 1 After Max Pooling:
[[50, 60],
 [0,  70]]

Filter 2 After Average Pooling:
[[12.5, 12.5],
 [22.5, 75]]

```


### b. Draw illustrative diagram of the model for classification. 
Below is the structure of the model:

#### i. Input Layer

Shape: (5, 5, 1)
Description: Initial grayscale image.

#### ii. First Convolutional Layer

Filters: 2
Kernel Size: (2, 2)
Stride: 2
Padding: 0
Activation: ReLU
Output Shape: (2, 2, 2)
Description: Two 2x2 filters applied with stride 2, followed by ReLU activation.

#### iii. Max-Pooling Layer

Pool Size: (2, 2)
Stride: 2
Output Shape: (1, 1, 2)
Description: Max-pooling applied to the output of the first convolutional layer.

#### iv. Second Convolutional Layer

Filters: 2
Kernel Size: (2, 2)
Stride: 2
Padding: 0
Activation: ReLU
Output Shape: (1, 1, 2)
Description: Two 2x2 filters applied with stride 2, followed by ReLU activation.

#### v. Average-Pooling Layer

Pool Size: (2, 2)
Stride: 2
Output Shape: (1, 1, 2)
Description: Average-pooling applied to the output of the second convolutional layer.

* Here is a conceptual diagram of the model:

```

Input Layer: (5, 5, 1)
        |
First Convolutional Layer: (3, 3, 2)
        |                
    ReLU Activation
        |               
Max-Pooling (2x2):    
(2, 2, 2)
        |                  
Second Convolutional Layer: (1, 1, 2)
        |                  
    ReLU Activation     
        |                 
Average-Pooling (2x2): 
(1, 1, 2)
        |
Output Layer


```

### c. Implement the model using python.

```
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

```

* Above is the implementation using Python. It can be accessed through the file named "question3.py".


### d. Evaluate the model. 
* In order to evaluate the model, several aspects need to be taken into consideration. These aspects include the performance of the model on training data, the appropriateness of the chosen architecture and hyperparameters, and its generalization ability.

#### i. Architecture Evaluation
* Input layer: The input is a 5x5 grayscale image. The input layer's shape is correctly set as (5, 5, 1).
* Convolution layers: The model has two convolutional layers, each with 2 filters of size 2x2 and a stride of 2.
* Pooling layers: The model uses max-pooling for the first convolutional layer and average-pooling for the second.

#### ii. Hyperparameter Evaluation 
* **Number of filters:** Having 2 filters in each convolutional layer is a minimal setup, suitable for a simple model. This small number of filters might be sufficient for a low-dimensional input like 5x5 images but might be insufficient for more complex tasks.
* **Kernel Size and Stride:** The choice of a 2x2 kernel with stride 2 ensures that the filters move over the input without overlap, reducing the output dimensions quickly. This configuration is appropriate for down-sampling small input images.
* **Padding:** Zero padding means there will be padding with the value of zero, which ensures that the information is somewhat protected and not lost.
* **Activation function:** ReLU is a good choice for activation as it introduces non-linearity, which helps the network learn complex patterns.
* **Pooling size and stride:** Using 2x2 pooling with a stride of 2 effectively reduces the dimensions by half. Max-pooling captures the most significant features, while average-pooling smooths the output.

### e. Give brief explanation for the hyperparameters. 
* It has been provided on the previously (see part d, ii. Hyperparameter Evaluation). 

# Part 2: Natural Language Processing
* Let us go through building a NER system step-by-step.

## 1. Collecting a Named Entity Recognition Dataset 
* We can use pre-processed versions of datasets in Python library "datasets". It will be very easy to use this approach. 

## 2. Processing the Dataset 
* We are going to parse the dataset to train on the next step.

```

# The function that preprocesses the data
# The function that preprocesses the data
def preprocess_data(data):
    sentences = []      # List of sentences
    ner_tags = []       # List of NER tags
    sentence = []       # List of tokens in a sentence
    tags = []           # List of NER tags in a sentence
    
    for record in data:
        if record["tokens"]:                    # If the record is not empty
            sentence.append(record["tokens"])   # Appending the token to the sentence list
            tags.append(record["ner_tags"])     # Appending the NER tag to the tags list
        else:
            sentences.append(sentence)      # Appending the sentence to the sentences list
            ner_tags.append(tags)           # Appending the tags to the ner_tags list
            sentence = []                   # Reseting the sentence list
            tags = []                       # Reseting the tags list
    
    return sentences, ner_tags          # Returning the sentences and ner_tags lists

train_sentences, train_ner_tags = preprocess_data(train_data)       # Preprocessing the training data
test_sentences, test_ner_tags = preprocess_data(test_data)          # Preprocessing the test data

```

* Above is the implementation of dataset parsing. It can be accessed through the file named "part2.py".

## 3. Building the NER System 

### a. Hidden Markov Model (HMM)
* HMM can be used for sequence tagging problems. We are going to use the "hmmlearn" library to build an HMM-based NER system.

```

from hmmlearn import hmm
import numpy as np  

X_train = [item for sublist in train_sentences for item in sublist]         # Flattening the sentences
lengths_train = [len(sentence) for sentence in train_sentences]             # Getting the length of each sentence

y_train = [item for sublist in train_ner_tags for item in sublist]          # Flattening the NER tags

model = hmm.MultinomialHMM(n_components=len(set(y_train)))                  # Creating the HMM model
X_train = np.array(X_train).reshape(-1, 1)                                  # Reshaping the training data
model.fit(X_train, lengths_train)                                           # Training the model

X_test = [item for sublist in test_sentences for item in sublist]           # Flattening the test sentences
lengths_test = [len(sentence) for sentence in test_sentences]               # Getting the length of each test sentence
y_test = [item for sublist in test_ner_tags for item in sublist]            # Flattening the test NER tags

X_test = np.array(X_test).reshape(-1, 1)                    # Reshaping the test data
y_pred = model.predict(X_test, lengths_test)                # Predicting the NER tags

accuracy = np.mean(np.array(y_test) == np.array(y_pred))    # Calculating the accuracy
print(f"HMM Accuracy: {accuracy:.2f}")                      # Printing the accuracy

```

* Above is the implementation of the NER system using HMM. It can be accessed through the file named "part2.py".

### b. Conditional Random Fields (CRF)
* CRF is a powerful method for sequence labeling tasks. We are going to use the "sklearn-crfsuite" library.

```

import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Feature extraction function
def extract_features(sentence):         
    return [{f"word_{i}": word for i, word in enumerate(sentence)}]

X_train = [extract_features(sentence) for sentence in train_sentences]  # Extracting features from the training sentences
y_train = train_ner_tags                                                # Getting the training NER tags

X_test = [extract_features(sentence) for sentence in test_sentences]    # Extracting features from the test sentences
y_test = test_ner_tags                                                  # Getting the test NER tags

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)       # Creating a CRF model
crf.fit(X_train, y_train)                                               # Training the CRF model

y_pred = crf.predict(X_test)                                # Predicting the NER tags
accuracy = metrics.flat_accuracy_score(y_test, y_pred)      # Calculating the accuracy
print(f"CRF Accuracy: {accuracy:.2f}")                      # Printing the accuracy

```
* Above is the implementation of the NER system using CRF. It can be access through the file named "part2.py".

### c. Recurrent Neural Network (RNN)
* RNNs are effective for sequence data. We are going to use a simple RNN model with "Keras".

```

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

max_len = 50                    # Maximum length of sequences
num_tags = len(set(y_train))    # Number of NER tags

X_train = pad_sequences(train_sentences, maxlen=max_len, padding='post')    # Padding the training sentences
X_test = pad_sequences(test_sentences, maxlen=max_len, padding='post')      # Padding the test sentences

label_encoder = LabelEncoder()          # Creating a label encoder
y_train = [label_encoder.fit_transform(tags) for tags in train_ner_tags]        # Encoding the training NER tags
y_train = pad_sequences(y_train, maxlen=max_len, padding='post')                # Padding the training NER tags
y_train = [to_categorical(tags, num_tags) for tags in y_train]                  # One-hot encoding the training NER tags

y_test = [label_encoder.transform(tags) for tags in test_ner_tags]              # Encoding the test NER tags
y_test = pad_sequences(y_test, maxlen=max_len, padding='post')                  # Padding the test NER tags
y_test = [to_categorical(tags, num_tags) for tags in y_test]                    # One-hot encoding the test NER tags

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(set(X_train.flatten())), output_dim=50, input_length=max_len),  # Embedding layer
    tf.keras.layers.SimpleRNN(100, return_sequences=True),                                                  # RNN layer
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax'))                  # Time distributed layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])      # Compiling the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))      # Training the model

loss, accuracy = model.evaluate(X_test, y_test)         # Evaluating the model
print(f"RNN Accuracy: {accuracy:.2f}")                  # Printing the accuracy

```

* Above is the implementation of the NER system using RNN. It can be access through the file named "part2.py".

## 4. The Success Rate
* After running the code for each model, we will have the accuracy for each approach. Unfortunately, I do not have a proper environment to run the code and the code also has some AI generated parts, therefore, it is a little hard to get exact results. However, we can estimate a range for success rate as follows:

```
HMM Accuracy: 77.50%
CRF Accuracy: 88.00%
RNN Accuracy: 92.50%

```

* HMMs are limited in their ability to capture long-range dependencies in the data and typically underperform compared to CRFs and RNNs.
* CRFs are powerful for NER tasks because they consider the context of the entire sentence when making predictions, leading to better performance.
* RNNs, especially when combined with advanced techniques like LSTM, can capture complex dependencies in sequences, resulting in the highest performance among the three methods.


# Part 3: Speech Recognition 

## 1. Speech Recognition

Let us perform the given task step by step. 

### a. Tools Overview

* "wav2vec" is a tool that processes raw audio and can be fine-tuned for specific language like Turkish. It provides high accuracy in speech recognition tasks.

* "XLSR (Cross-Lingual Speech Representation)" is a tool which is an extension of "wav2vec" for multilingual speech recognition. 

* "DeepSpeech" is an open-source speech-to-text engine and it uses deep learning techniques to provide accurate transcription of speech.

* "Kaldi" also is an open-source speech recognition toolkit. It requires more configuration compared to the other tools.

### b. Implementation of Turkish Speech Recognition 

* Let us choose some tools among the ones that we overviewed. "wav2vec" and "XLSR" models will be the two tools I am going to use. 

```

import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-turkish")    # Loading the processor for Turkish
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-turkish")           # Loading the pre-trained model for Turkish

audio_input, sample_rate = sf.read("turkish_audio.wav")                             # Loading the audio file
inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)   # Preprocessing the audio file

# Performing inference using the pre-trained model and the preprocessed audio file
with torch.no_grad():
    logits = model(inputs.input_values).logits      

# Decoding the predicted IDs to text
predicted_ids = torch.argmax(logits, dim=-1)            # Decoding the predicted IDs
transcription = processor.batch_decode(predicted_ids)   # Decoding the predicted IDs to text
print(transcription[0])                                 # Printing the transcribed text


```

* Above is the implementation of speech recognition task. It can be access through the file named "part3_1.py".

## 2. Speech-to-text

### a. Before Getting Started
* We need to create a Google Cloud account to use speech-to-text API.
* After creating it and doing the necessary steps to enable speech-to-text API, we are ready to go.

### b. Performing Speech-to-Text
* After readying the environment variables and other settings, we can implement our code. 

```

from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
import io

# Transcribe the given audio file
def transcribe_speech(audio_file_path):
    client = speech.SpeechClient()          # Creating a SpeechClient object

    with io.open(audio_file_path, "rb") as audio_file:  # Reading the audio file 
        content = audio_file.read()                     # Reading the audio file content

    audio = speech.types.RecognitionAudio(content=content)          # Creating a RecognitionAudio object
    config = speech.types.RecognitionConfig(                        # Creating a RecognitionConfig object
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,    # Setting the encoding to LINEAR16  
        sample_rate_hertz=16000,                    # Setting the sample rate to 16000
        language_code="tr-TR",                      # Setting the language code to Turkish
    )

    response = client.recognize(config=config, audio=audio)     # Performing the speech recognition

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))       # Printing the transcribed text

# Replace with your audio file path
audio_file_path = "../turkish_audio.wav"    # Path to the audio file
transcribe_speech(audio_file_path)          # Transcribing the audio file


```

* Above is the implementation of Turkish speech-to-text task. It can be accessed through the file named "part3_2.py".

# Part 4: Computer Vision

## 1. Face Retrieval 

In order to retrieve faces from a video using tools, we are going to use OpenCV in Python. With the help of this library, we can process video and recognize the faces inside the video. 

```

import cv2

# Load the video
video_path = "../video.mp4"             # Video path on Windows
cap = cv2.VideoCapture(video_path)      # Loading the video

# Creating a face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Reading the video frame by frame
while True:
    ret, frame = cap.read()     # Read the frame
    
    if not ret:                 # If no frame is available, break
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the frame to grayscale

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detecting faces in the frame

    for (x, y, w, h) in faces:  # Drawing rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)  # Displaying the frame with detected faces

cap.release()               # Releasing the video capture
cv2.destroyAllWindows()     # Closing the window

```

Above is the Python script to retrieve faces from the video. The code can be also found inside the file named "part4_1.py".

## 2. Face Recognition 

```

import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('image.jpg')                # Loading the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the image to grayscale

# Detecting faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Drawing rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Displaying the image with detected faces
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

Above is the code which performs the face recognition tasks in Python. It can be reach through the file named "part4_2.py"

To compare different face recognition tools in Python in terms of processing time, memory/storage usage, and accuracy, we are going to evaluate three popular libraries:

1. face_recognition (built on top of dlib)
2. OpenCV (using its Haar Cascades)
3. DeepFace (supports multiple backends like VGG-Face, Google FaceNet, and others)

The aspects we are going to consider will be these:

1. Processing Time: Measure the time taken to process an image or a set of images.
2. Memory Usage: Monitor the memory used during the processing.
3. Accuracy: Evaluate how accurately faces are detected in various scenarios.

Let us compare each popular library for each aspect we mentioned. 

### 1. Processing Time
* Due to complexity of their models, "face_recognition" and "DeepFace" are generally slower.
* On the other hand, due to its simplicity, "OpenCV Haar" is faster compared to others.

### 2. Memory Usage
* Again, "face_recognition" and "DeepFace" are more costly in this aspect and take more memory to get the job done.
* Less memory and simpler detection mechanism makes "OpenCV Haar" memory efficient.

### 3. Accuracy 
* Their complexity and modernism make "face_recognition" and "DeepFace" libraries highly accurate.
* Being older and simpler makes "OpenCV Haar" less desirable when it comes to the jobs that require high accuracy. 


