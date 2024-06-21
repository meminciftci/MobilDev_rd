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


# --------------------------------- HMM ----------------------------------

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

# ----------------------------------------------------------------------------

# --------------------------------- CRF --------------------------------------

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


# ----------------------------------------------------------------------------

# --------------------------------- RNN -------------------------------------

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

# ----------------------------------------------------------------------------

