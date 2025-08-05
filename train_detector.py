import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("osr_features.csv")  # or your combined features.csv
X = df.drop("label", axis=1)
y = df["label"].map({"real": 0, "fake": 1})  # numeric target

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

# Evaluate with 5-fold CV
f1_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
print(f"5-fold CV F1 scores: {f1_scores}")
print(f"Mean F1: {f1_scores.mean():.4f}")

# Fit final model on all data
clf.fit(X, y)
joblib.dump(clf, "deepfake_detector_osr.pkl")
print("Model saved as deepfake_detector_osr.pkl")
