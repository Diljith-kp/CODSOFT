# spam_detection.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("--- Data Ready (First 5 Rows) ---")
print(df.head())
print("\nNumber of Good Mail (0) and Junk Mail (1):")
print(df['label'].value_counts())

def preprocess_text_simplified(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_message'] = df['message'].apply(preprocess_text_simplified)

print("\n--- Messages Cleaned (Original vs. Cleaned) ---")
print(df[['message', 'cleaned_message']].head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X = tfidf_vectorizer.fit_transform(df['cleaned_message'])

y = df['label']

print(f"\n--- Text Converted to Numbers ---")
print(f"Your messages are now {X.shape[0]} rows and {X.shape[1]} numbers (features) each.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- Data Split for Practice and Test ---")
print(f"Practice mail for robot (Training): {X_train.shape[0]} messages.")
print(f"Test mail for robot (Testing): {X_test.shape[0]} messages.")

print("\n--- Training Robot Brains and Checking Their Report Cards ---")

print("\n### Brain 1: Naive Bayes (Simple Counter) ###")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision (Junk Mail - Correctly Flagged as Junk): {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall (Junk Mail - Junk Mail Caught): {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1-Score (Junk Mail - Balance of Precision/Recall): {f1_score(y_test, y_pred_nb):.4f}")
print("Confusion Matrix (How many were correctly/incorrectly sorted):\n", confusion_matrix(y_test, y_pred_nb))

print("\n### Brain 2: Logistic Regression (Line Drawer) ###")
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision (Junk Mail - Correctly Flagged as Junk): {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall (Junk Mail - Junk Mail Caught): {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score (Junk Mail - Balance of Precision/Recall): {f1_score(y_test, y_pred_lr):.4f}")
print("Confusion Matrix (How many were correctly/incorrectly sorted):\n", confusion_matrix(y_test, y_pred_lr))

print("\n### Brain 3: Support Vector Machine (SVM) (Best Separator) ###")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision (Junk Mail - Correctly Flagged as Junk): {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall (Junk Mail - Junk Mail Caught): {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score (Junk Mail - Balance of Precision/Recall): {f1_score(y_test, y_pred_svm):.4f}")
print("Confusion Matrix (How many were correctly/incorrectly sorted):\n", confusion_matrix(y_test, y_pred_svm))

warnings.filterwarnings('default')

import joblib

joblib.dump(svm_model, 'best_spam_model.pkl')
print("\nBest model (SVM) saved as 'best_spam_model.pkl'")

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")

print("\n--- Testing the Best Model with New Messages ---")

new_messages = [
    "Congratulations! You've won a FREE prize! Click here!",
    "Hey, how are you doing today? Let's catch up soon.",
    "URGENT! Your account has been compromised. Verify now!",
    "Can you pick up milk on your way home?",
    "You have won a lottery ticket. Claim your cash prize"
]

loaded_model = joblib.load('best_spam_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("\nPredicting on new messages:")
for msg in new_messages:
    cleaned_msg = preprocess_text_simplified(msg)
    msg_numeric = loaded_vectorizer.transform([cleaned_msg])
    prediction = loaded_model.predict(msg_numeric)[0]
    result = "Junk Mail (Spam)" if prediction == 1 else "Good Mail (Ham)"
    print(f"Message: '{msg}'\nPrediction: {result}\n")