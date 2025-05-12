import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# Загрузка данных
X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

# Усредним временное окно (из 3D сделаем 2D)
X_flat = X.mean(axis=1)  # (samples, features)

# Обучим LightGBM
model = lgb.LGBMClassifier()
model.fit(X_flat, y)

# Получим важность признаков
importances = model.feature_importances_

# Список имён признаков (должен совпадать с твоим порядком в create_dataset)
feature_names = [
    'RSI', 'MACD', 'MACD_signal', 'MA_7', 'MA_21',
    'Volatility', 'Daily_Return', 'Volume_Change',
    'Open_Close_Diff', 'High_Low_Pct',
    'Close_t-1', 'Close_t-2',
    'Momentum_3', 'Momentum_7',
    'Bollinger_width',
    'OBV'
]

# Отсортируем по убыванию
sorted_idx = np.argsort(importances)[::-1]

print("📊 Важность признаков:")
for i in sorted_idx:
    print(f"{feature_names[i]:<20} → {importances[i]}")

# Построим график
plt.figure(figsize=(10, 5))
plt.barh(
    [feature_names[i] for i in sorted_idx],
    [importances[i] for i in sorted_idx]
)
plt.gca().invert_yaxis()
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()
