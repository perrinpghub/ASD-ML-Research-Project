import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# Load dataset
df_raw = pd.read_csv("asd_synthetic_dataset_v2.csv")
df = df_raw.copy()

# Encode categorical columns (NOT the target)
cat_cols = ["gender", "parent_education", "ethnicity", "jaundice", "family_ASD"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features / target
X_raw = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ---- FIX: convert diagnosis to numeric 0/1 (required for ROC/metrics) ----
# Handles values like: "ASD", "Not ASD", "not asd", etc.
y_str = y.astype(str).str.strip().str.lower()
mapping = {
    "asd": 1,
    "not asd": 0,
    "not_asd": 0,
    "non asd": 0,
    "non-asd": 0,
    "nonasd": 0,
    "0": 0,
    "1": 1,
}
y = y_str.map(mapping)

if y.isna().any():
    # show unknown labels to help debugging
    unknown = sorted(set(y_str[y.isna()].unique().tolist()))
    raise ValueError(f"Unknown diagnosis labels found in CSV: {unknown}")

y = y.astype(int)

# Feature selection (RFE top 12)
rfe_model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
rfe = RFE(rfe_model, n_features_to_select=12)
X = rfe.fit_transform(X_raw, y)
selected_features = X_raw.columns[rfe.support_].tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE on training only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Model: SVM + NB soft voting
svm = SVC(probability=True, random_state=RANDOM_STATE)
nb = GaussianNB()
model = VotingClassifier(estimators=[("svm", svm), ("nb", nb)], voting="soft")

model.fit(X_train_bal, y_train_bal)

# Save test data (needed for plotting + Appendix B)
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# Save everything needed for inference
bundle = {
    "model": model,
    "scaler": scaler,
    "rfe": rfe,
    "selected_features": selected_features,
    "encoders": encoders,
    "raw_columns": list(X_raw.columns),
    "cat_cols": cat_cols,
    "random_state": RANDOM_STATE,
}
joblib.dump(bundle, "asd_screening_model.joblib")

# ---- Print evaluation (useful for Appendix B) ----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("— Saved model bundle to: asd_screening_model.joblib")
print("— Saved test files: X_test.pkl, y_test.pkl")
print("— Selected features:", selected_features)
print("\n--- Evaluation on Test Set ---")
print("ROC-AUC:", auc)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))