import os
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow
import joblib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# --- Assuming your config.py defines these ---
from ..config import (
    PREPROCESSED_DATA_DIR,  # Not directly used in this script but often part of a project
    TARGETS,                # Not directly used
    TARGET3,                # Used for TARGET_COLUMN_NAME
    PREDICTOR,              # Not directly used
    MODELS_DIR,             # Used for local saving
    TYPE_PREPROCESSED_DIR   # Used for data loading
)
# --- End Config Import ---

# --- MLflow Configuration ---
mlflow.set_tracking_uri("http://localhost:5000") # Your MLflow tracking server
mlflow.set_experiment("rest-mex-xgboost_hyperopt_gpu_experiment")

# --- Static Configuration for this script run ---
TARGET_COLUMN_NAME = TARGET3
FEATURES_FILENAME_PREFIX = "w2v"
TRAIN_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_train.csv"
TEST_DATA_FILENAME = f"{FEATURES_FILENAME_PREFIX}_test.csv"
RANDOM_SEED = 42
MAX_HYPEROPT_EVALS = 10 # Number of Hyperopt trials

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
    X_train_full = train_df[feature_columns_train].copy() # Use .copy() to avoid SettingWithCopyWarning
    y_train_full_original = train_df[target_col].copy()
    X_test_final = test_df[[col for col in test_df.columns if col != target_col]].copy()
    y_test_final_original = test_df[target_col].copy()

    label_encoder = LabelEncoder()
    y_train_full_encoded = label_encoder.fit_transform(y_train_full_original)
    y_test_final_encoded = label_encoder.transform(y_test_final_original)
    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} classes for target '{target_col}': {list(label_encoder.classes_)}")
    return X_train_full, y_train_full_encoded, X_test_final, y_test_final_encoded, num_classes, label_encoder

# --- XGBoost Model Creation ---
def create_xgboost_model(params: dict, num_classes: int, random_state: int) -> xgb.XGBClassifier:
    """
    Creates an XGBClassifier model based on the provided params.
    Handles early_stopping_rounds as a constructor parameter for XGBoost >= 2.0.0.
    """
    model_params = {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": params.get("eval_metric", "mlogloss"),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "gamma": params["gamma"],
        "reg_alpha": params.get("reg_alpha", 0),
        "reg_lambda": params.get("reg_lambda", 1),
        "random_state": random_state,
        "n_jobs": -1, # Usually good for CPU, GPU usage is controlled by tree_method/device
        "tree_method": "gpu_hist", # Should be 'gpu_hist' for GPU
    }

    if params["tree_method"] == 'gpu_hist':
        model_params["device"] = "cuda" # Explicitly set device for gpu_hist

    # For XGBoost 2.0.0+ (including 3.0.2), early_stopping_rounds is a constructor param
    early_stopping_rounds_val = params.get("early_stopping_rounds")
    if early_stopping_rounds_val is not None and early_stopping_rounds_val > 0:
        model_params["early_stopping_rounds"] = int(early_stopping_rounds_val)

    print(f"Creating XGBClassifier model with params: {model_params}")
    model = xgb.XGBClassifier(**model_params)
    return model

# --- Model Training ---
def train_xgboost_model(model: xgb.XGBClassifier, X_train_data: pd.DataFrame, y_train_data: np.ndarray,
                        X_val_data: pd.DataFrame, y_val_data: np.ndarray):
    """
    Trains the XGBoost model. For XGBoost >= 2.0.0, early stopping is configured
    at model initialization. The eval_set is still passed to fit().
    """
    print(f"\nStarting XGBClassifier model training...")
    fit_params = {
        "eval_set": [(X_val_data, y_val_data)],
        "verbose": False # XGBoost's own verbosity during training
    }
    # If early_stopping_rounds was set in constructor, XGBoost will use eval_set[0] by default
    # No need to pass early_stopping_rounds to fit() for XGBoost 2.x+

    model.fit(X_train_data, y_train_data, **fit_params)
    print("Model training finished.")
    return model

# --- Model Evaluation ---
def evaluate_xgboost_model(model: xgb.XGBClassifier, X_data: pd.DataFrame, y_data_encoded: np.ndarray, target_names: list):
    print("\nEvaluating XGBClassifier model...")
    y_pred = model.predict(X_data)
    accuracy = accuracy_score(y_data_encoded, y_pred)
    # Ensure target_names are strings if they come from label_encoder.classes_
    target_names_str = [str(name) for name in target_names]
    report = classification_report(y_data_encoded, y_pred, target_names=target_names_str, output_dict=True, zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_data_encoded, y_pred, target_names=target_names_str, zero_division=0))
    return accuracy, report

# --- Hyperopt Search Space Definition ---
space = {
    "n_estimators": hp.quniform("n_estimators", 100, 800, 50),
    "max_depth": hp.quniform("max_depth", 3, 8, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "gamma": hp.uniform("gamma", 0.0, 0.7),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-5), np.log(1.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-5), np.log(1.0)),
    "early_stopping_rounds": hp.quniform("early_stopping_rounds", 10, 50, 5),
    "tree_method": "gpu_hist", # Fixed for GPU training
    # "device": "cuda" # This is handled in create_xgboost_model if tree_method is gpu_hist
    "eval_metric": hp.choice("eval_metric", ["mlogloss", "merror"]),
}

# --- Global variables for objective function ---
X_train_opt_g, y_train_opt_encoded_g, X_val_opt_g, y_val_opt_encoded_g = None, None, None, None
g_num_classes_g, g_label_encoder_g, g_target_names_list_g = None, None, None

# --- Objective Function for Hyperopt ---
def objective(params):
    global g_num_classes_g, g_target_names_list_g, X_train_opt_g, y_train_opt_encoded_g, X_val_opt_g, y_val_opt_encoded_g

    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    # early_stopping_rounds is now passed to constructor, but keep it as int if used for logic
    params["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 0))


    with mlflow.start_run(nested=True) as run: # MLflow automatically names runs if not specified
        run_id = run.info.run_id
        # Construct a more descriptive run name tag (optional, MLflow UI shows params anyway)
        # Default MLflow run names are usually fine for hyperopt trials.
        # mlflow.set_tag("mlflow.runName", f"xgb_trial_{params['max_depth']}_{params['learning_rate']:.4f}_{run_id[:6]}")
        print(f"\n--- Hyperopt Trial --- MLflow Run ID: {run_id}")
        print(f"Attempting with parameters: {params}")
        mlflow.log_params(params)

        model = create_xgboost_model(params, num_classes=g_num_classes_g, random_state=RANDOM_SEED)
        
        train_xgboost_model(
            model, X_train_opt_g, y_train_opt_encoded_g,
            X_val_data=X_val_opt_g, y_val_data=y_val_opt_encoded_g
            # early_stopping_rounds_config is now part of model constructor params
        )

        val_accuracy, val_report_dict = evaluate_xgboost_model(model, X_val_opt_g, y_val_opt_encoded_g, target_names=g_target_names_list_g)
        
        mlflow.log_metric("validation_accuracy", val_accuracy)
        for class_or_avg, metrics_dict in val_report_dict.items():
            if isinstance(metrics_dict, dict):
                for metric_name, metric_value in metrics_dict.items():
                    safe_metric_name = f"val_{class_or_avg.replace(' ', '_')}_{metric_name}".lower()
                    mlflow.log_metric(safe_metric_name, metric_value)
        
        return {'loss': -val_accuracy, 'status': STATUS_OK, 'params_ H': params, 'run_id_H': run_id}


# --- Main Execution Block ---
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print(f"XGBoost version being used: {xgb.__version__}") # Good to log this

    print(f"Loading data for target: {TARGET_COLUMN_NAME} using {FEATURES_FILENAME_PREFIX} features.")
    try:
        X_train_full, y_train_full_encoded, X_test_final, y_test_final_encoded, num_classes, label_encoder_main = \
            load_and_prepare_data(
                target_preprocessed_dir=TYPE_PREPROCESSED_DIR,
                train_filename=TRAIN_DATA_FILENAME,
                test_filename=TEST_DATA_FILENAME,
                target_col=TARGET_COLUMN_NAME
            )
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure preprocessing scripts have run.")
        exit()
    except Exception as e:
        print(f"Data loading/preparation error: {e}")
        exit()

    X_train_opt, X_val_opt, y_train_opt_encoded_g_init, y_val_opt_encoded_g_init = train_test_split(
        X_train_full, y_train_full_encoded, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_full_encoded
    )
    print(f"Data split for Hyperopt: Train shape: {X_train_opt.shape}, Validation shape: {X_val_opt.shape}")

    globals()['X_train_opt_g'] = X_train_opt
    globals()['y_train_opt_encoded_g'] = y_train_opt_encoded_g_init
    globals()['X_val_opt_g'] = X_val_opt
    globals()['y_val_opt_encoded_g'] = y_val_opt_encoded_g_init
    globals()['g_num_classes_g'] = num_classes
    globals()['g_label_encoder_g'] = label_encoder_main
    globals()['g_target_names_list_g'] = list(label_encoder_main.classes_)

    print(f"\n--- Starting Hyperopt Search (max {MAX_HYPEROPT_EVALS} evals) ---")
    trials = Trials()
    with mlflow.start_run(run_name=f"XGB_Hyperopt_Search_{FEATURES_FILENAME_PREFIX}") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("max_hyperopt_evals", MAX_HYPEROPT_EVALS)
        mlflow.set_tag("mlflow.note.content", "Parent run for XGBoost (GPU) hyperparameter optimization using Hyperopt.")
        mlflow.log_param("xgboost_version", xgb.__version__)


        best_hyperparams_from_fmin = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=MAX_HYPEROPT_EVALS,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED)
        )
        
        # fmin returns space indices for 'hp.choice', need to map back
        # For 'tree_method', if it's hp.choice, best_hyperparams_from_fmin['tree_method'] will be an index.
        # We need the actual parameters from the best trial object.
        best_trial_params = trials.best_trial['result']['params_ H']
        
        mlflow.log_params({f"best_trial_{k}": v for k,v in best_trial_params.items()})
        best_trial_loss = trials.best_trial['result']['loss']
        mlflow.log_metric("best_validation_accuracy", -best_trial_loss)
        best_trial_run_id = trials.best_trial['result']['run_id_H']
        mlflow.set_tag("best_trial_run_id", best_trial_run_id)

    print(f"\n--- Hyperopt Search Finished ---")
    print(f"Best trial parameters found by Hyperopt: {best_trial_params}")
    print(f"Best validation accuracy (from best trial): {-trials.best_trial['result']['loss']:.4f}")
    
    # Ensure integer types for the final model from best_trial_params
    final_best_params = best_trial_params.copy()
    final_best_params["n_estimators"] = int(final_best_params["n_estimators"])
    final_best_params["max_depth"] = int(final_best_params["max_depth"])
    final_best_params["early_stopping_rounds"] = int(final_best_params.get("early_stopping_rounds", 0))


    print("\n--- Training Final Model with Best Hyperparameters on Full Training Data ---")
    with mlflow.start_run(run_name=f"XGB_BestModel_GPU_{FEATURES_FILENAME_PREFIX}") as final_run:
        final_run_id = final_run.info.run_id
        mlflow.set_tag("source_hyperopt_parent_run_id", parent_run_id)
        mlflow.set_tag("source_hyperopt_best_trial_run_id", best_trial_run_id)
        mlflow.log_params({f"final_best_{k}": v for k,v in final_best_params.items()})
        mlflow.log_param("final_training_on_full_data", True)
        mlflow.log_param("xgboost_version", xgb.__version__)

        final_model = create_xgboost_model(final_best_params, num_classes=g_num_classes_g, random_state=RANDOM_SEED)
        
        final_model = train_xgboost_model(
            final_model, X_train_full, y_train_full_encoded,
            X_val_data=X_test_final, y_val_data=y_test_final_encoded
            # Early stopping config is now part of final_best_params used in create_xgboost_model
        )

        print("\n--- Evaluating Final Best Model on Test Set ---")
        test_accuracy, report_dict = evaluate_xgboost_model(final_model, X_test_final, y_test_final_encoded, target_names=g_target_names_list_g)

        mlflow.log_metric("final_test_accuracy", test_accuracy)
        for class_or_avg, metrics_dict in report_dict.items():
            if isinstance(metrics_dict, dict):
                for metric_name, metric_value in metrics_dict.items():
                    safe_metric_name = f"final_test_{class_or_avg.replace(' ', '_')}_{metric_name}".lower()
                    mlflow.log_metric(safe_metric_name, metric_value)
        
        mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path=f"model_final_best_xgb_gpu",
            # registered_model_name=f"xgb_gpu_best_{TARGET3.lower()}_{FEATURES_FILENAME_PREFIX}" # Optional
        )
        print(f"Final XGBoost GPU model logged to MLflow.")

        le_filename = f"label_encoder_{TARGET3.lower()}_final_best_gpu.pkl"
        le_local_path = MODELS_DIR / le_filename
        joblib.dump(g_label_encoder_g, le_local_path)
        mlflow.log_artifact(le_local_path, artifact_path="label_encoder_final_gpu")
        print(f"LabelEncoder saved to {le_local_path} and logged to MLflow.")

        model_local_save_path = MODELS_DIR / f"xgb_gpu_{TARGET3.lower()}_model_{FEATURES_FILENAME_PREFIX}_final_best.pkl"
        joblib.dump(final_model, model_local_save_path)
        print(f"Final model also saved locally to: {model_local_save_path}")

    print("\nHyperparameter optimization and final model training complete.")