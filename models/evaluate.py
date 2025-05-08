
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

def evaluate_model(model, X_test, y_test, config):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
        if config['task'] == 'classification' and preds.ndim > 1:
            preds = preds.argmax(axis=1)

    os.makedirs("results", exist_ok=True)

    if config['task'] == 'regression':
        mae = mean_absolute_error(y_test, preds)
        print("MAE:", mae)

        plt.figure()
        plt.plot(preds[:100], label="Prediction")
        plt.plot(y_test[:100], label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual")
        plt.savefig("results/prediction_vs_actual.png")
        print("Saved plot to results/prediction_vs_actual.png")

    else:
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig("results/confusion_matrix.png")
        print("Saved plot to results/confusion_matrix.png")
