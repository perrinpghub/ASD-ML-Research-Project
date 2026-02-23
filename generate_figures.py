import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report

MODEL_BUNDLE_FILE = "asd_screening_model.joblib"
XTEST_FILE = "X_test.pkl"
YTEST_FILE = "y_test.pkl"

def convert_labels_to_numeric(y):
    """
    Converts y to 0/1 if it contains strings like 'ASD'/'Not ASD'.
    If already numeric, returns unchanged.
    """
    try:
        # If it's numpy array / pandas series
        y_arr = np.array(y)

        # numeric already?
        if np.issubdtype(y_arr.dtype, np.number):
            return y_arr.astype(int)

        # convert strings
        y_str = np.array([str(v).strip().lower() for v in y_arr])
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
        y_num = np.array([mapping.get(v, None) for v in y_str], dtype=object)
        if any(v is None for v in y_num):
            unknown = sorted(set([y_str[i] for i, v in enumerate(y_num) if v is None]))
            raise ValueError(f"Unknown labels in y_test: {unknown}")

        return y_num.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert labels to numeric: {e}")

# Load bundle + model
bundle = joblib.load(MODEL_BUNDLE_FILE)
model = bundle["model"]

# Load test data
X_test = joblib.load(XTEST_FILE)
y_test = joblib.load(YTEST_FILE)

# Safety: ensure numeric labels
y_test = convert_labels_to_numeric(y_test)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics (for Appendix B if needed)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Evaluation on Test Set ---")
print("ROC-AUC:", roc_auc)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ---------------------------
# Figure: Confusion Matrix
# ---------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Non-ASD (0)", "ASD (1)"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=15)
plt.yticks(tick_marks, classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ---------------------------
# Figure: ROC Curve
# ---------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_plot = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_plot:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

print("\n— Figures saved:")
print(" - confusion_matrix.png")
print(" - roc_curve.png")