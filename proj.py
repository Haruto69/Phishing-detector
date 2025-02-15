import re
import joblib
import numpy as np
import ctypes

# Load the saved model and vectorizer
classifier = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load the shared library
trie_lib = ctypes.CDLL('./libtest2.dll')  # Use './libtest2.so' on Linux/MacOS

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

# Preprocess Emails
def preprocess_email(email):
    email = email.lower()
    email = re.sub(r'[^\w\s]', '', email)  # Remove punctuation
    email = re.sub(r'\d+', '', email)     # Remove numbers
    return email

# Extract features using Trie
def get_trie_features(email):
    c_email = email.encode('utf-8')

    # Allocate memory for the feature vector
    feature_vector = (ctypes.c_int * MAX_KEYWORDS)()

    # Call the C function to get the feature vector
    trie_lib.getFeatureVector(root, c_email, feature_vector, MAX_KEYWORDS)

    # Convert the feature vector to a Python list
    return list(feature_vector)

# Function to predict email
def predict_email(email):
    # Preprocess the email
    processed_email = preprocess_email(email)

    # Vectorize the email
    email_vectorized = vectorizer.transform([processed_email]).toarray()

    # Get the Trie features for the email
    email_trie_features = get_trie_features(processed_email)
    email_trie_features_padded = email_trie_features + [0] * (MAX_KEYWORDS - len(email_trie_features))

    # Combine the vectorized and Trie features
    email_combined = np.hstack((email_vectorized, [email_trie_features_padded]))

    # Make the prediction using the pre-trained classifier
    prediction = classifier.predict(email_combined)

    return prediction[0]

# Function to predict whether an email is a scam or not
def predict_scam(email):
    result = predict_email(email)
    return result
