import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    confusion_matrix
)
from models.lstm_model import LSTMPredictor
import yaml
import matplotlib.pyplot as plt
import os

def evaluate_model(config):
    os.makedirs("results", exist_ok=True)

    # Load test data
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    
    # Evaluate
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()

            probs = torch.sigmoid(outputs)

            y_true.extend(y_batch.cpu().numpy())
            y_probs.extend(outputs.cpu().numpy())
            y_pred.extend((probs > 0.5).float().cpu().numpy())
    print(f"Min prob: {np.min(y_probs):.4f}, Max: {np.max(y_probs):.4f}, Mean: {np.mean(y_probs):.4f}")
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save metrics to file
    with open("results/metrics.txt", "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:200], label='Actual')
    plt.plot(y_pred[:200], label='Predicted', alpha=0.7)
    plt.title("Sample Predictions vs Actual")
    plt.legend()
    plt.savefig("results/predictions.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(y_probs, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.savefig("results/probability_distribution.png")
    plt.close()



if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    evaluate_model(config)