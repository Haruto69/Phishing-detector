import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset from CSV file
df = pd.read_csv('emails.csv')

# Separate emails and labels
emails = df['text'].tolist()
labels = df['label'].tolist()

# Preprocess Emails
def preprocess_email(email):
    email = email.lower()
    email = re.sub(r'[^\w\s]', '', email)  # Remove punctuation
    email = re.sub(r'\d+', '', email)     # Remove numbers
    return email

processed_emails = [preprocess_email(email) for email in emails]

# Trie Definition
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.count += 1

    def get_feature_vector(self, words):
        """Generate a feature vector of keyword counts from an email."""
        feature_vector = []
        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    break
                node = node.children[char]
            feature_vector.append(node.count if "children" in dir(node) else 0)
        return feature_vector

# Initialize Trie and Add Words
trie = Trie()
for email in processed_emails:
    for word in email.split():
        trie.insert(word)

# Extract Features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(processed_emails)
X_trie_features = [trie.get_feature_vector(email.split()) for email in processed_emails]

# Pad Trie feature vectors
max_trie_features = max(len(f) for f in X_trie_features)
X_trie_features_padded = np.array([
    f + [0] * (max_trie_features - len(f)) for f in X_trie_features
])

# Combine Features
X_combined = np.hstack((X_vectorized.toarray(), X_trie_features_padded))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.3, random_state=42)

# Train the Classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the Model
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to Check Email
def check_email():
    while True:
        new_email = input("\nEnter an email to check if it's spam or not: ")
        processed_new_email = preprocess_email(new_email)
        new_vectorized = vectorizer.transform([processed_new_email]).toarray()
        new_trie_features = trie.get_feature_vector(processed_new_email.split())
        new_trie_features_padded = new_trie_features + [0] * (max_trie_features - len(new_trie_features))
        new_combined = np.hstack((new_vectorized, [new_trie_features_padded]))
        prediction = classifier.predict(new_combined)
        print(f"Prediction: {'Spam' if prediction[0] == 'spam' else 'Not Spam'}")

        # Ask if the user wants to continue
        choice = input("\nDo you want to check another email? (yes/no): ").strip().lower()
        if choice not in ('yes', 'y'):
            print("Exiting the program.")
            break

# Run the interactive loop
check_email()
