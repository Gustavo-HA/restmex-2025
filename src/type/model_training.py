import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Useful if you want to split again for validation

from ..config import (
    PREPROCESSED_DATA_DIR,
    TARGETS,        
    TARGET3,        
    PREDICTOR,      
    MODELS_DIR,     
    TYPE_PREPROCESSED_DIR 
)

# --- Configuration for this specific model ---
TARGET_COLUMN_NAME = TARGET3 # Directly use TARGET3 from config.py
FEATURES_FILENAME_PREFIX = "tfidf" # Or "w2v" depending on which features you want to use
TRAIN_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_train.csv"
TEST_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_test.csv"
# Construct model save path using MODELS_DIR and TARGET3 name
MODEL_SAVE_PATH = MODELS_DIR / f"mlp_{TARGET3.lower()}_model_{FEATURES_FILENAME_PREFIX}.h5"

# MLP Hyperparameters (mapping from your scikit-learn example)
HIDDEN_LAYER_SIZES = (256, 128, 64)
ACTIVATION_FUNCTION = 'relu'
OPTIMIZER_LEARNING_RATE = 1e-3
L2_REGULARIZATION_ALPHA = 1e-4
EPOCHS = 300
BATCH_SIZE = 32
RANDOM_SEED = 42
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 30

# --- GPU Setup ---
def setup_gpu():
    """
    Configures TensorFlow to use available GPUs and allows memory growth.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {len(gpus)}")
            # Optional: If you have multiple GPUs and want to restrict to one
            # tf.config.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU devices found. Training will run on CPU.")

# --- Data Loading and Preparation ---
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

# --- MLP Model Definition ---
def create_mlp_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Creates a Multi-Layer Perceptron (MLP) model using Keras.
    """
    model = tf.keras.Sequential()

    # Input layer and first hidden layer
    model.add(tf.keras.layers.Dense(
        units=HIDDEN_LAYER_SIZES[0],
        activation=ACTIVATION_FUNCTION,
        input_shape=input_shape,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION_ALPHA),
        name="hidden_layer_1"
    ))

    # Additional hidden layers
    for i, units in enumerate(HIDDEN_LAYER_SIZES[1:]):
        model.add(tf.keras.layers.Dense(
            units=units,
            activation=ACTIVATION_FUNCTION,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION_ALPHA),
            name=f"hidden_layer_{i+2}"
        ))

    # Output layer
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation='softmax', # Use softmax for multi-class classification
        name="output_layer"
    ))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=OPTIMIZER_LEARNING_RATE),
        loss='sparse_categorical_crossentropy', # For integer labels (0, 1, 2...)
        metrics=['accuracy']
    )

    model.summary()
    return model

# --- Model Training ---
def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: np.ndarray,
                X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> tf.keras.callbacks.History:
    """
    Trains the Keras MLP model.
    """
    callbacks = []
    if ENABLE_EARLY_STOPPING:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ))
    
    validation_data = None
    validation_split_param = 0.0 # Default to 0, if explicit validation_data is provided
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    elif ENABLE_EARLY_STOPPING: # If early stopping is enabled and no explicit val set, use validation_split
        validation_split_param = 0.1 # Keras will use 10% from training data for validation

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
        validation_split=validation_split_param,
        callbacks=callbacks,
        verbose=1
    )
    print("Model training finished.")
    return history

# --- Model Evaluation ---
def evaluate_model(model: tf.keras.Model, X_test: pd.DataFrame, y_test: np.ndarray):
    """
    Evaluates the trained Keras MLP model on the test set.
    """
    print("\nEvaluating model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Set Random Seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 2. Setup GPU
    setup_gpu()

    # 3. Load and Prepare Data
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


    # 4. Create Model
    model = create_mlp_model(input_shape, num_classes)

    # 5. Train Model
    history = train_model(model, X_train, y_train_encoded)

    # 6. Evaluate Model
    loss, accuracy = evaluate_model(model, X_test, y_test_encoded)

    # 7. Save Model
    # Ensure the directory for saving models exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    import joblib
    joblib.dump(label_encoder, MODELS_DIR / f"label_encoder_{TARGET3.lower()}.pkl")
    print(f"LabelEncoder saved to {MODELS_DIR / f'label_encoder_{TARGET3.lower()}.pkl'}")