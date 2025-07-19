import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

print("--- Checking/Downloading NLTK resources ---")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' corpus...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading 'wordnet' corpus...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading 'omw-1.4' corpus...")
    nltk.download('omw-1.4')
print("NLTK resources ready.\n")


def load_data(filepath, has_genre=True):
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory as the script.")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if has_genre:
                if len(parts) == 4:
                    data.append({
                        'id': parts[0],
                        'title': parts[1],
                        'genre': parts[2],
                        'description': parts[3]
                    })
                else:
                    print(f"Warning: Skipping malformed line in {filepath} (expected 4 parts, got {len(parts)}): {line.strip()}")
            else:
                if len(parts) == 3:
                    data.append({
                        'id': parts[0],
                        'title': parts[1],
                        'description': parts[2]
                    })
                else:
                    print(f"Warning: Skipping malformed line in {filepath} (expected 3 parts, got {len(parts)}): {line.strip()}")
    return pd.DataFrame(data)

print("--- Loading train_data.txt ---")
try:
    train_df = load_data('train_data.txt', has_genre=True)
    print(f"Successfully loaded {len(train_df)} training samples.")
    print("Training data head:\n", train_df.head())
except FileNotFoundError as e:
    print(e)
    exit()

print("\n--- Loading test_data.txt (for prediction) ---")
try:
    test_df = load_data('test_data.txt', has_genre=False)
    print(f"Successfully loaded {len(test_df)} test samples for prediction.")
    print("Test data head:\n", test_df.head())
except FileNotFoundError as e:
    print(e)
    exit()

print("\n--- Loading test_data_solution.txt (for evaluation) ---")
try:
    test_solution_df = load_data('test_data_solution.txt', has_genre=True)
    test_solution_df = test_solution_df.set_index('id')
    test_df = test_df.set_index('id')
    print(f"Successfully loaded {len(test_solution_df)} test solution samples.")
    print("Test solution data head:\n", test_solution_df.head())
except FileNotFoundError as e:
    print(e)
    exit()

common_ids = list(set(test_df.index).intersection(set(test_solution_df.index)))
test_df = test_df.loc[common_ids].copy()
test_solution_df = test_solution_df.loc[common_ids].copy()
y_true_test = test_solution_df['genre']

print(f"\nAligned test data for evaluation: {len(test_df)} samples.\n")


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

print("--- Applying text preprocessing to training data descriptions ---")
train_df['processed_description'] = train_df['description'].apply(preprocess_text)
print("Preprocessing complete for training data.")

print("\n--- Applying text preprocessing to test data descriptions ---")
test_df['processed_description'] = test_df['description'].apply(preprocess_text)
print("Preprocessing complete for test data.\n")


tfidf_vectorizer = TfidfVectorizer(max_features=7000, min_df=5, max_df=0.7)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_description'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_description'])
y_train = train_df['genre']

print(f"TF-IDF Vectorization complete. Number of features (words): {X_train_tfidf.shape[1]}\n")


print("--- Training Naive Bayes Classifier ---")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Accuracy:", accuracy_score(y_true_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_true_test, y_pred_nb, zero_division=0))
print("-" * 50 + "\n")

print("--- Training Logistic Regression Classifier ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_true_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_true_test, y_pred_lr, zero_division=0))
print("-" * 50 + "\n")

print("--- Training Linear Support Vector Machine (LinearSVC) Classifier ---") # Removed "This should be faster"
svm_model = LinearSVC(random_state=42, max_iter=2000)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("LinearSVC Accuracy:", accuracy_score(y_true_test, y_pred_svm))
print("LinearSVC Classification Report:\n", classification_report(y_true_test, y_pred_svm, zero_division=0))
print("-" * 50 + "\n")


chosen_model = svm_model
print(f"--- Generating predictions for test_data.txt using {chosen_model.__class__.__name__} ---")

test_df['predicted_genre'] = chosen_model.predict(X_test_tfidf)

output_lines = []
for index, row in test_df.iterrows():
    output_lines.append(f"{index} ::: {row['title']} ::: {row['predicted_genre']} ::: {row['description']}")

output_filepath = 'test_data_predictions.txt'
with open(output_filepath, 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"Predictions saved to '{output_filepath}'")
print("\n--- Sample of generated predictions (first 5 lines from the output file) ---")
with open(output_filepath, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 4:
            break
print("\n--- All tasks completed successfully! ---")