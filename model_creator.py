import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import ctypes

# Load dataset
df = pd.read_csv('emails.csv')  # Replace with your dataset path
emails = df['text']
labels = df['label']

# Preprocess Emails
def preprocess_email(email):
    email = email.lower()
    email = re.sub(r'[^\w\s]', '', email)  # Remove punctuation
    email = re.sub(r'\d+', '', email)     # Remove numbers
    return email

processed_emails = emails.apply(preprocess_email)

# Load the shared library
trie_lib = ctypes.CDLL('./libtest2.dll')  # Use './libtest2.so' for Linux/MacOS

# Define constants
MAX_KEYWORDS = 100

# Define the Trie Node structure
class TrieNode(ctypes.Structure):
    _fields_ = [("children", ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)) * 26),
                ("count", ctypes.c_int)]

NODE = ctypes.POINTER(TrieNode)

# Load the C functions
trie_lib.createNode.restype = NODE
trie_lib.insert.argtypes = [NODE, ctypes.c_char_p]
trie_lib.getFeatureVector.argtypes = [
    NODE,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]

# Initialize the Trie
root = trie_lib.createNode()

# Populate the Trie with email words
for email in processed_emails:
    for word in email.split():
        trie_lib.insert(root, word.encode('utf-8'))

# Extract features using Trie
def get_trie_features(email):
    c_email = email.encode('utf-8')

    # Allocate memory for the feature vector
    feature_vector = (ctypes.c_int * MAX_KEYWORDS)()

    # Call the C function to get the feature vector
    trie_lib.getFeatureVector(root, c_email, feature_vector, MAX_KEYWORDS)

    # Convert the feature vector to a Python list
    return list(feature_vector)

# Vectorize text features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(processed_emails)

# Extract Trie features
X_trie_features = [get_trie_features(email) for email in processed_emails]

# Pad Trie feature vectors
X_trie_features_padded = np.array([
    features + [0] * (MAX_KEYWORDS - len(features)) for features in X_trie_features
])

# Combine features
X_combined = np.hstack((X_vectorized.toarray(), X_trie_features_padded))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(classifier, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'.")
