import argparse
import yaml
import torch
import os

from data.prepare_data import download_data, prepare_all_data

from models.train_gru import train_model as train_gru_model
from models.train_mlp import train_model as train_mlp_model
from models.train_lstm import train_model as train_lstm_model

from models.evaluate_gru import evaluate_model as evaluate_gru_model
from models.evaluate_mlp import evaluate_model as evaluate_mlp_model
from models.evaluate_lstm import evaluate_model as evaluate_lstm_model


def main():
    parser = argparse.ArgumentParser(description="Stock Market Prediction Pipeline")
    parser.add_argument('--download', action='store_true', help='Download raw data')
    parser.add_argument('--prepare', action='store_true', help='Prepare processed data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model type from config
    model_type = config['model']['type'].lower()

    if args.download or args.all:
        print("\n=== Downloading data ===")
        download_data(
            config['data']['tickers'],
            config['data']['start_date'],
            config['data']['end_date'])

    if args.prepare or args.all:
        print("\n=== Preparing data ===")
        prepare_all_data(config)

    if args.train or args.all:
        print("\n=== Training model ===")
        if model_type == "lstm":
            train_lstm_model(config)
        elif model_type == "gru":
            train_gru_model(config)
        elif model_type == "mlp":
            train_mlp_model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    if args.evaluate or args.all:
        print("\n=== Evaluating model ===")
        if model_type == "lstm":
            evaluate_lstm_model(config)
        elif model_type == "gru":
            evaluate_gru_model(config)
        elif model_type == "mlp":
            evaluate_mlp_model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    main()