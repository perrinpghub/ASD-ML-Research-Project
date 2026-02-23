import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

RANDOM_STATE = 42

# ==========================================================
# IMPORTANT NOTE (for final defense / report honesty):
# This dataset is SYNTHETIC / SIMULATION data created for a
# coursework demo (NOT clinical/real patient data).
# ==========================================================

# Load dataset
df = pd.read_csv("asd_synthetic_dataset_v2.csv")

# Encode categorical variables
label_cols = ['gender', 'parent_education', 'ethnicity', 'jaundice', 'family_ASD', 'diagnosis']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
X_raw = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Feature selection (RFE -> top 12 indicators)
rfe_model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
rfe = RFE(rfe_model, n_features_to_select=12)
X = rfe.fit_transform(X_raw, y)
selected_features = X_raw.columns[rfe.support_].tolist()
print("Selected features (RFE=12):", selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data balancing with SMOTE (train only)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Define classifiers
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=250)
svm = SVC(probability=True, random_state=RANDOM_STATE)
nb = GaussianNB()
lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
knn = KNeighborsClassifier()

# Voting classifiers
voting1 = VotingClassifier(estimators=[('dt', dt), ('rf', rf)], voting='soft')
voting2 = VotingClassifier(estimators=[('svm', svm), ('nb', nb)], voting='soft')
voting3 = VotingClassifier(estimators=[('lr', lr), ('knn', knn)], voting='soft')

# Fit and evaluate
for name, clf in [('DT+RF', voting1), ('SVM+NB', voting2), ('LR+KNN', voting3)]:
    clf.fit(X_train_bal, y_train_bal)
    y_pred = clf.predict(X_test)

    print(f"\n{name} Classifier Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        pass
