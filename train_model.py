import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. File paths
# ==============================

DATASET_PATH = "Resume dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "resume_model.pkl")


# ==============================
# 2. Load dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded successfully!")
print(f"Total resumes: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# ==============================
# 3. Check required columns
# ==============================

required_columns = ["category", "job_title", "Text"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# ==============================
# 4. Clean data
# ==============================

df = df.dropna(subset=["Text", "category"])

df["Text"] = df["Text"].astype(str)
df["category"] = df["category"].astype(str)

# Remove empty resumes
df = df[df["Text"].str.strip() != ""]

print(f"\nResumes after cleaning: {len(df)}")


# ==============================
# 5. Features and target
# ==============================

X = df["Text"]
y = df["category"]


# ==============================
# 6. Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==============================
# 7. Create ML Pipeline
# ==============================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ==============================
# 8. Train model
# ==============================

print("\nTraining AI model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==============================
# 9. Evaluate model
# ==============================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==============================
# 10. Create models folder
# ==============================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# 11. Save trained model
# ==============================

joblib.dump(model, MODEL_PATH)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")
print(f"Model location: {MODEL_PATH}")


# ==============================
# 12. Test with sample resume
# ==============================

sample_resume = """
Python developer with experience in machine learning,
data analysis, pandas, numpy, scikit-learn,
deep learning and artificial intelligence.
"""

prediction = model.predict([sample_resume])

print("\nSample Resume Prediction:")
print(prediction[0])

print("\nTraining process completed successfully!")