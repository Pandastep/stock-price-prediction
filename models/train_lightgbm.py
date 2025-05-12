import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Загрузка данных
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Преобразуем: берём последний временной срез
X_train = X_train[:, -1, :]
X_val = X_val[:, -1, :]
X_test = X_test[:, -1, :]

# Объединяем train и val
X_train_full = np.concatenate([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# Обучение модели
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=100,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_full, y_train_full)

# Предсказания
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

# Метрики
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_probs)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== LightGBM Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Сохраняем метрики
os.makedirs("results", exist_ok=True)
with open("results/metrics_lightgbm.txt", "w") as f:
    f.write("LightGBM Metrics\n================\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ROC AUC: {roc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

# Важность признаков
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=20)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("results/feature_importance_lightgbm.png")
plt.close()
