import numpy as np
import predictions
import tensorflow as tf
import pandas as pd
import re
import nltk
import nltk.tokenize.punkt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
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

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_liar)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts_liar).toarray()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train the K-NN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Training Accuracy: {accuracy:.2f}")
print("Training Classification Report:\n", classification_report(y_test, y_pred))


# Load test data and predict
liar_df_test_data = pd.read_csv('test.tsv', header=None, sep='\t',
                      names=["id", "label", "statement", "subject", "speaker", "speakerjobtitle", "stateinfo",
                             "partyaffiliation", "barelytruecounts", "falsecounts", "halftruecounts",
                             "mostlytruecounts", "pantsonfirecounts", "context"])

# test statements
test_texts_liar = liar_df_test_data['statement'].values  # Assuming 'statement' column contains the text
x=0
for stmt in test_texts_liar:
    test_texts_liar[x] = preprocess_text(stmt)
    x = x+1

# Encode test labels
liar_df_test_data['label'] = liar_df_test_data['label'].map(label_map)
test_labels_liar = liar_df_test_data['label'].values
y_test_actual = label_encoder.fit_transform(test_labels_liar)

# Vectorize test texts using the same vectorizer
X_test_data = vectorizer.transform(test_texts_liar).toarray()

# Make predictions on test data
y_test_pred = knn.predict(X_test_data)
test_accuracy = accuracy_score(y_test_actual, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:\n", classification_report(y_test_actual, y_test_pred))

# test printing
#for i in range(100):
#    print("Test number " + str(i+1) + " prediction result is " + str(y_test_pred[i]) + " should be " + str(y_test_actual[i]))