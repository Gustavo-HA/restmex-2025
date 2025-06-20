import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
import numpy as np
from ..config import (
    PREPROCESSED_DATA_DIR,
    TARGETS,
    TARGET3,
    PREDICTOR,
    MODELS_DIR,
    TYPE_PREPROCESSED_DIR
)
from joblib import dump
import argparse

parser = argparse.ArgumentParser(description="Train Logistic Regression model with specified feature representation.")
parser.add_argument(
    "--features",
    type=str,
    required=True,
    help="Feature representation prefix (e.g., 'w2v', 'tfidf', etc.)"
)
args = parser.parse_args()
if args.features not in {"w2v", "tfidf"}:
    raise ValueError("Invalid feature prefix. Only 'w2v' and 'tfidf' are allowed.")

def get_logistic_regression_model():
    """
    Returns a Logistic Regression model wrapped in OneVsOneClassifier.
    """
    # Create a Logistic Regression model
    lr_model = LogisticRegression(
        max_iter=1000,  
        random_state=42, 
        solver='liblinear'
    )
    
    ovr_model = OneVsOneClassifier(lr_model)
    
    return ovr_model


def load_and_prepare_data(target_preprocessed_dir: Path, train_filename: str, test_filename: str, target_col: str):
    """
    Loads preprocessed data and separates features (X) from the target (y).
    Encodes string labels to integers.
    """
    train_path = target_preprocessed_dir / train_filename
    test_path = target_preprocessed_dir / test_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data file: '{train_path}'. Ensure preprocessing created it.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test data file: '{test_path}'. Ensure preprocessing created it.")

    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features (X) and target (y)
    feature_columns_train = [col for col in train_df.columns if col != target_col]
    feature_columns_test = [col for col in test_df.columns if col != target_col]

    X_train = train_df[feature_columns_train]
    y_train = train_df[target_col]
    X_test = test_df[feature_columns_test]
    y_test = test_df[target_col]

    # For sparse_categorical_crossentropy, labels should be integers (0, 1, 2...)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} classes for target '{target_col}': {label_encoder.classes_}")

    return X_train, y_train_encoded, X_test, y_test_encoded, num_classes, label_encoder

    

# --- Configuration for this specific model ---
TARGET_COLUMN_NAME = TARGET3
FEATURES_FILENAME_PREFIX = args.features
TRAIN_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_train.csv"
TEST_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_test.csv"
# Construct model save path using MODELS_DIR and TARGET3 name
MODEL_SAVE_PATH = MODELS_DIR / f"lr_{TARGET3.lower()}_model_{FEATURES_FILENAME_PREFIX}.pkl"

if __name__ == "__main__":
    #Set Random Seeds for reproducibility
    np.random.seed(42)

    # Load and Prepare Data
    print(f"Loading data for target: {TARGET_COLUMN_NAME} using {FEATURES_FILENAME_PREFIX} features.")
    try:
        X_train, y_train_encoded, X_test, y_test_encoded, num_classes, label_encoder = \
            load_and_prepare_data(
                target_preprocessed_dir=TYPE_PREPROCESSED_DIR, # Use TYPE_PREPROCESSED_DIR directly
                train_filename=TRAIN_DATA_FILENAME,
                test_filename=TEST_DATA_FILENAME,
                target_col=TARGET_COLUMN_NAME
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run your cleaning and preprocessing scripts to generate the necessary CSVs.")
        exit()
    except ValueError as e:
        print(f"Data preparation error: {e}")
        exit()

    # Get input shape from the training features
    input_shape = (X_train.shape[1],)
    print(f"Input feature shape: {input_shape}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    model = get_logistic_regression_model()

    model.fit(X_train, y_train_encoded)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    dump(model, MODEL_SAVE_PATH)
    print("Model saved successfully.")

    dump(label_encoder, MODELS_DIR / f"label_encoder_{TARGET3.lower()}.pkl")
    print(f"LabelEncoder saved to {MODELS_DIR / f'label_encoder_{TARGET3.lower()}.pkl'}")