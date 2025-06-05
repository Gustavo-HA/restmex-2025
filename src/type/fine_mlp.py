import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import mlflow
import joblib # For saving label encoder

# Assuming your config.py defines these correctly
from ..config import (
    PREPROCESSED_DATA_DIR,
    TARGETS,
    TARGET3,
    PREDICTOR,
    MODELS_DIR,
    TYPE_PREPROCESSED_DIR
)


# --- MLflow Configuration ---
mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
mlflow.set_experiment("rest-mex-mlp_training_experiment_v2") # Set/Create your MLflow experiment name

# --- Static Configuration for this script run ---
TARGET_COLUMN_NAME = TARGET3
FEATURES_FILENAME_PREFIX = "w2v" # Or "tf-idf" depending on features
TRAIN_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_train.csv"
TEST_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_test.csv"
RANDOM_SEED = 42

# --- MLP Configurations to Experiment With ---
MLP_CONFIGURATIONS = [
    {
        "run_name": "MLP_default_short_epoch",
        "hidden_layer_sizes": (256, 128, 64),
        "activation_function": 'relu',
        "optimizer_learning_rate": 1e-3,
        "l2_regularization_alpha": 1e-4,
        "epochs": 300, # Reduced for quick testing, adjust as needed
        "batch_size": 32,
        "enable_early_stopping": True,
        "early_stopping_patience": 10,
    },
    {
        "run_name": "MLP_larger_net_smaller_lr",
        "hidden_layer_sizes": (512, 256, 128),
        "activation_function": 'relu',
        "optimizer_learning_rate": 1e-3,
        "l2_regularization_alpha": 1e-4,
        "epochs": 300, # Reduced for quick testing
        "batch_size": 64,
        "enable_early_stopping": True,
        "early_stopping_patience": 15,
    },
    {
        "run_name": "MLP_fewer_layers_no_l2",
        "hidden_layer_sizes": (128, 64),
        "activation_function": 'relu',
        "optimizer_learning_rate": 1e-3,
        "l2_regularization_alpha": 0.0, # No L2
        "epochs": 300, # Reduced for quick testing
        "batch_size": 32,
        "enable_early_stopping": False, # Test without early stopping
        "early_stopping_patience": 0, # Irrelevant if not enabled
    }
]


# --- GPU Setup ---
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {len(gpus)}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU devices found. Training will run on CPU.")

# --- Data Loading and Preparation ---
def load_and_prepare_data(target_preprocessed_dir: Path, train_filename: str, test_filename: str, target_col: str):
    train_path = target_preprocessed_dir / train_filename
    test_path = target_preprocessed_dir / test_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data file: '{train_path}'.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test data file: '{test_path}'.")

    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_columns_train = [col for col in train_df.columns if col != target_col]
    X_train = train_df[feature_columns_train]
    y_train = train_df[target_col]
    X_test = test_df[[col for col in test_df.columns if col != target_col]] # Ensure same feature set
    y_test = test_df[target_col]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} classes for target '{target_col}': {list(label_encoder.classes_)}")
    return X_train, y_train_encoded, X_test, y_test_encoded, num_classes, label_encoder

# --- MLP Model Definition ---
def create_mlp_model(input_shape: tuple, num_classes: int,
                     hidden_layer_sizes_config: tuple,
                     activation_function_config: str,
                     l2_regularization_alpha_config: float,
                     optimizer_learning_rate_config: float) -> tf.keras.Model:
    model = tf.keras.Sequential(name="MLP_Sequential")
    # Keras Input layer for explicit input shape definition
    model.add(tf.keras.layers.Input(shape=input_shape, name="input_layer"))

    for i, units in enumerate(hidden_layer_sizes_config):
        model.add(tf.keras.layers.Dense(
            units=units,
            activation=activation_function_config,
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_alpha_config) if l2_regularization_alpha_config > 0 else None,
            name=f"hidden_layer_{i+1}"
        ))
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        name="output_layer"
    ))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimizer_learning_rate_config),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

# --- Model Training ---
def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: np.ndarray,
                X_val: pd.DataFrame = None, y_val: np.ndarray = None,
                epochs_config: int = 100, # Default epochs
                batch_size_config: int = 32, # Default batch size
                enable_early_stopping_config: bool = True, # Default early stopping
                early_stopping_patience_config: int = 30) -> tf.keras.callbacks.History: # Default patience
    callbacks = []
    if enable_early_stopping_config:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience_config,
            restore_best_weights=True,
            verbose=1
        ))
    
    validation_data = None
    validation_split_param = 0.0
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    elif enable_early_stopping_config: # Use validation_split only if early stopping is on and no X_val/y_val
        validation_split_param = 0.1 

    print(f"\nStarting model training with: epochs={epochs_config}, batch_size={batch_size_config}")
    history = model.fit(
        X_train, y_train,
        epochs=epochs_config,
        batch_size=batch_size_config,
        validation_data=validation_data,
        validation_split=validation_split_param,
        callbacks=callbacks,
        verbose=1 # Can set to 2 for less output per epoch or 0 for silent
    )
    print("Model training finished.")
    return history

# --- Model Evaluation ---
def evaluate_model(model: tf.keras.Model, X_test: pd.DataFrame, y_test: np.ndarray):
    print("\nEvaluating model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

# --- Main Execution Block ---
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    setup_gpu()

    print(f"Loading data for target: {TARGET_COLUMN_NAME} using {FEATURES_FILENAME_PREFIX} features.")
    try:
        X_train, y_train_encoded, X_test, y_test_encoded, num_classes, label_encoder = \
            load_and_prepare_data(
                target_preprocessed_dir=TYPE_PREPROCESSED_DIR,
                train_filename=TRAIN_DATA_FILENAME,
                test_filename=TEST_DATA_FILENAME,
                target_col=TARGET_COLUMN_NAME
            )
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure preprocessing scripts have run.")
        exit()
    except Exception as e: # Catch other potential data errors
        print(f"Data loading/preparation error: {e}")
        exit()

    input_shape = (X_train.shape[1],)
    print(f"Input feature shape: {input_shape}")

    for config in MLP_CONFIGURATIONS:
        current_run_name = config.get("run_name", "mlp_default_run")
        print(f"\n--- Starting MLflow Run for Configuration: {current_run_name} ---")

        with mlflow.start_run(run_name=current_run_name) as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")
            print(f"Parameters for this run: {config}")

            # Log parameters
            mlflow.log_params({
                "hidden_layer_sizes": str(config["hidden_layer_sizes"]), # MLflow prefers strings for list/tuple params
                "activation_function": config["activation_function"],
                "optimizer_learning_rate": config["optimizer_learning_rate"],
                "l2_regularization_alpha": config["l2_regularization_alpha"],
                "epochs_configured": config["epochs"],
                "batch_size": config["batch_size"],
                "enable_early_stopping": config["enable_early_stopping"],
                "early_stopping_patience": config["early_stopping_patience"],
                "random_seed": RANDOM_SEED,
                "features_type": FEATURES_FILENAME_PREFIX,
                "target_column": TARGET_COLUMN_NAME
            })

            # Create and Train Model
            print("Creating MLP model...")
            model = create_mlp_model(
                input_shape=input_shape,
                num_classes=num_classes,
                hidden_layer_sizes_config=config["hidden_layer_sizes"],
                activation_function_config=config["activation_function"],
                l2_regularization_alpha_config=config["l2_regularization_alpha"],
                optimizer_learning_rate_config=config["optimizer_learning_rate"]
            )

            history = train_model(
                model, X_train, y_train_encoded, # Pass X_test, y_test_encoded for validation_data if desired
                X_val=X_test, y_val=y_test_encoded, # Using test set as validation for this example
                epochs_config=config["epochs"],
                batch_size_config=config["batch_size"],
                enable_early_stopping_config=config["enable_early_stopping"],
                early_stopping_patience_config=config["early_stopping_patience"]
            )

            # Evaluate Model
            test_loss, test_accuracy = evaluate_model(model, X_test, y_test_encoded)

            # Log metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            # Log training history metrics if needed (e.g., final training loss/accuracy)
            if history.history:
                mlflow.log_metric("final_train_loss", history.history['loss'][-1])
                mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
                if 'val_loss' in history.history:
                    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
                    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
            
            # Log model
            # The artifact_path is relative to the run's artifact directory
            mlflow.keras.log_model(model, artifact_path=f"model_{current_run_name}")
            print(f"Keras model logged to MLflow for run: {current_run_name}")

            # Save and log LabelEncoder locally and to MLflow
            le_filename = f"label_encoder_{TARGET3.lower()}_{current_run_name}.pkl"
            le_local_path = MODELS_DIR / le_filename
            joblib.dump(label_encoder, le_local_path)
            mlflow.log_artifact(le_local_path, artifact_path="label_encoder")
            print(f"LabelEncoder saved to {le_local_path} and logged to MLflow.")

            # Optional: Save model locally with a unique name for this run
            model_local_save_path = MODELS_DIR / f"mlp_{TARGET3.lower()}_model_{FEATURES_FILENAME_PREFIX}_{current_run_name}.keras"
            # model.save(model_local_save_path) # Already logged with mlflow.keras.log_model, local save is optional
            # print(f"Model also saved locally to: {model_local_save_path}")

            print(f"--- Finished MLflow Run for Configuration: {current_run_name} ---")

    print("\nAll MLP configurations trained and logged to MLflow.")