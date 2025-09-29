import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# --- Configuration ---
DATA_FILE = "housing_data.csv"
MODEL_ONNX_FILE = "house_price_model.onnx"
SCALER_FILE = "scalers.pkl"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # apple silicon
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")   
else:
    DEVICE = torch.device("cpu")
print(f"Using Device : {DEVICE}")



def load_and_preprocess_data(data_path, test_size : float = 0.2, validation_size : float = 0.2, random_state : int = 21):
    try:
       df = pd.read_csv(data_path)
    except FileNotFoundError:
       print(f"Error : the file '{data_path} was not found.") 
       return None, None, None, None, None, None, None, None

    features = ["square_footage", "bedrooms", "bathrooms"]
    target = "price_thousands"

    X = df[features].values 
    y = df[[target]].values

    # Scaler Data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_scaled, test_size = test_size, random_state = random_state)

    val_to_train_ratio = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = test_size, random_state = random_state)


    # convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype = torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype = torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype = torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype = torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype = torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype = torch.float32).to(DEVICE)


    print("data loaded and pre-processed successfully.")

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, feature_scaler, target_scaler


class HousePricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features : sq. footage, number of bedrooms, number of bathrooms
        self.fc1 = nn.Linear(3, 64)         
        # Hidden Layer
        self.fc2 = nn.Linear(64, 32)
        # Output Layer
        self.fc3 = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, epochs = 250, lr = 0.001, patience = 50, min_delta = 0.0001):
    pass

if __name__ == "__main__":

    # command line flags
    parser = argparse.ArgumentParser(
        description = "A simple neural network for housing price prediction using PyTorch + ONNX"
    )

    parser.add_argument(
        "--data_path", type = str, default = DATA_FILE, help = f"Path to the CSV file (default : {DATA_FILE})"
    )

    parser.add_argument(
        "--train", action = "store_true", help = "Train the model and save as an ONNX file."
    )

    parser.add_argument(
        "--predict", action = "store_true", help = "Load an ONNX model and make predcition(s)"
    )

    parser.add_argument(
        "--model_path", type = str, default = MODEL_ONNX_FILE, help = "Path to save / load the ONNX model to / from"
    )

    parser.add_argument(
        "--scaler_path", type = str, default = SCALER_FILE, help = "Path to save / load the MinMaxScaler objects"
    )

    parser.add_argument(
        "--input_features", type = str, help = "Comma separated input features for prediction (eg. '1000, 1, 5'). To be used with --predict"
    )

    parser.add_argument(
        "--epochs", type = int, default = 250, help = "Number of training iterations (defaults to 250). Use with --train"
    )

    parser.add_argument(
        "--lr", type = float, default = 0.001, help = "Model learning rate for training (defaults to 0.001)"
    )

    parser.add_argument(
        "--patience", type = int, default = 50, help = "Number of epochs to wait for improvement before enforcing early stopping (defaults to 50). Use with --train"
    )

    parser.add_argument(
        "--min_delta", type = float, default = 0.0001, help = "Minimum change in test loss to qualify as an 'improvement' from an early stopping perspective (defaults to 0.0001). Use with --train"
    )


    args = parser.parse_args()

    if args.train :
        print("--- Training Mode ---")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = load_and_preprocess_data(args.data_path)
        if X_train is not None:
            model = HousePricePredictor().to(DEVICE)
    elif args.predict:
        print("--- Prediction Mode ---")

    else:
        print("Error : Please specify either --train or --predict")
        parser.print_help()