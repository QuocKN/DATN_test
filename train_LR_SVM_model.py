import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# load data
X = np.load("X.npy")
y = np.load("y.npy")

# chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale (RẤT QUAN TRỌNG cho SVM / Logistic)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # ========================
# # 1. Logistic Regression
# # ========================
# logistic = LogisticRegression(max_iter=1000)
# logistic.fit(X_train, y_train)

# y_pred_log = logistic.predict(X_test)

# print("=== Logistic Regression ===")
# print("Accuracy:", accuracy_score(y_test, y_pred_log))
# print(classification_report(y_test, y_pred_log))

# ========================
# 2. SVM
# ========================
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_svm)

labels = ["Noise", "Drone"]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format='d')

plt.title("Confusion Matrix (SVM + DINOv2)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("confusion_matrix.png", dpi=300)  # lưu ảnh chất lượng cao
plt.show()