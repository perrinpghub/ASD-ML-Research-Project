import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

df_raw = pd.read_csv("asd_synthetic_dataset_v2.csv")
df = df_raw.copy()

# Encode categorical
cat_cols = ["gender", "parent_education", "ethnicity", "jaundice", "family_ASD"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X_raw = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Feature selection (RFE top 12)
rfe_model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
rfe = RFE(rfe_model, n_features_to_select=12)
X = rfe.fit_transform(X_raw, y)
selected_features = X_raw.columns[rfe.support_].tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Best model: SVM + NB soft voting
svm = SVC(probability=True, random_state=RANDOM_STATE)
nb = GaussianNB()
model = VotingClassifier(estimators=[("svm", svm), ("nb", nb)], voting="soft")

model.fit(X_train_bal, y_train_bal)

# Save everything needed for inference
bundle = {
    "model": model,
    "scaler": scaler,
    "rfe": rfe,
    "selected_features": selected_features,
    "encoders": encoders,
    "raw_columns": list(X_raw.columns),
    "cat_cols": cat_cols,
}
joblib.dump(bundle, "asd_screening_model.joblib")
print("Saved model bundle to asd_screening_model.joblib")
print("Selected features:", selected_features)