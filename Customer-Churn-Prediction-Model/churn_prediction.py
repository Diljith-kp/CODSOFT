import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # This line helps Matplotlib display plots in a new window, especially on Windows.
import seaborn as sns

#%% 
# --- 1. Load Data ---
try:
    df = pd.read_csv('Churn_Modelling.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: 'Churn_Modelling.csv' not found. Make sure it's in the same folder.")
    exit()

#%% 
# --- 2. Prepare Data ---
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

X = df.drop(columns=['Exited'])
y = df['Exited']

categorical_features = ['Geography', 'Gender']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

#%% 
# --- 3. Train and Evaluate Models ---
results = {}

def train_and_evaluate(model_pipeline, name):
    print(f"\n--- {name} ---")
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'y_prob': y_prob
    }

#%%
# a. Logistic Regression
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])
train_and_evaluate(pipeline_lr, "Logistic Regression")

#%%
# b. Decision Tree Classifier
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(random_state=42))])
train_and_evaluate(pipeline_dt, "Decision Tree")

#%%
# c. Random Forest Classifier
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])
train_and_evaluate(pipeline_rf, "Random Forest")

#%%
# d. Gradient Boosting Classifier
pipeline_gb = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(random_state=42))])
train_and_evaluate(pipeline_gb, "Gradient Boosting")

print("\n--- Model training and evaluation complete. See output above for results. ---")

#%%
# --- PLOTTING SECTION ---
print("\n--- Generating ROC Curve Comparison Plot ---")
plt.figure(figsize=(10, 7))
for model_name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['ROC-AUC']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

#%%
print("\n--- Generating Feature Importance Plot for Gradient Boosting ---")
fitted_preprocessor = pipeline_gb.named_steps['preprocessor']
fitted_classifier = pipeline_gb.named_steps['classifier']

numerical_feature_names_transformed = numerical_features
categorical_feature_names_transformed = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

all_feature_names = numerical_feature_names_transformed + categorical_feature_names_transformed.tolist()

feature_importances = fitted_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Top 10 Feature Importances (from Gradient Boosting) ---")
print(importance_df.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Feature Importances for Churn Prediction (Gradient Boosting)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
