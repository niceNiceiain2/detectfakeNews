import numpy as np
import predictions
import tensorflow as tf
import pandas as pd
import re
import nltk
import nltk.tokenize.punkt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

liar_df = pd.read_csv('train.tsv', header=None, sep='\t',
                      names=["id","label","statement","subject","speaker","speakerjobtitle","stateinfo",
                             "partyaffiliation","barelytruecounts","falsecounts","halftruecounts",
                              "mostlytruecounts","pantsonfirecounts","context"])

# Map the labels to binary
label_map = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 0,
    "mostly-true": 1,
    "true": 1
}

# Define tokenizer and padding parameters
vocab_size = 10000       # Limit vocabulary size to the 10,000 most frequent words
max_length = 54         # Define a maximum length for padding
embedding_dim = 16       # Embedding dimension

# Apply the mapping to create binary labels
liar_df['label'] = liar_df['label'].map(label_map)
texts_liar = liar_df['statement'].values  # Assuming 'statement' column contains the text
labels_liar = liar_df['label'].values

# Initialize the Tokenizer with the defined vocabulary size and OOV token
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

x=0
for stmt in texts_liar:
    texts_liar[x] = preprocess_text(stmt)
    x = x+1

# Fit the tokenizer on the statements (our input texts)
tokenizer.fit_on_texts(texts_liar)

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(texts_liar)

# Pad the sequences to ensure uniform length across the dataset
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Print shapes and an example for verification
print("Shape of padded sequences:\n", padded_sequences.shape)
print("Example padded sequence: \n", padded_sequences[0])

# Build the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(padded_sequences, labels_liar, epochs=10, validation_split=0.2, batch_size=32)

print("This is the model after training")
model.summary()

print(model)

liar_df_test_data = pd.read_csv('test.tsv', header=None, sep='\t',
                      names=["id", "label", "statement", "subject", "speaker", "speakerjobtitle", "stateinfo",
                             "partyaffiliation", "barelytruecounts", "falsecounts", "halftruecounts",
                             "mostlytruecounts", "pantsonfirecounts", "context"])

liar_test_expected = [0] * 3000
liar_test_results = [0] * 3000
label_encoder = LabelEncoder()
liar_df_test_data['label'] = liar_df_test_data['label'].map(label_map)
for i in range(1200):
    # Predict using the trained model
    y = liar_df_test_data.iloc[i]
    stmt = y['statement']
    stmt = preprocess_text(stmt)
    tokenizer.fit_on_texts(stmt)
    sequences = tokenizer.texts_to_sequences([stmt])[0]
    sequences = pad_sequences([sequences], maxlen=max_length, padding='post', truncating='post')
    x = model.predict(sequences, verbose=0)[0][0]
    #print("Test number " + str(i+1) + " result is " + str(x) + " should be " + str(y['label']))
    liar_test_expected[i] = y['label']
    if x >= 0.5:
        #print("  This news is True")
        liar_test_results[i] = 1
        #print("Test number " + str(i+1) + " result is 1 should be " + str(y['label']))
    else:
        #print("  This news is False")
        liar_test_results[i] = 0
        #print("Test number " + str(i + 1) + " result is 0 should be " + str(y['label']))


test_accuracy = accuracy_score(liar_test_expected, liar_test_results)
print(f"Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:\n", classification_report(liar_test_expected, liar_test_results))