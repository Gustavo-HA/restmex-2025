from pathlib import Path
import json

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ..config import (
    PREPROCESSED_DATA_DIR,
    PREDICTOR,
    TARGET1,
    INTERIM_DATA_DIR
)

from .dataset import RMDataset

# --- Si no usas config.py, define las constantes aquí ---
RAW_DATA_FILE = INTERIM_DATA_DIR / "train_set_Polarity.csv"
TARGET_COLUMN = TARGET1
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

PROCESSED_DATA_DIR = PREPROCESSED_DATA_DIR / "polarity"



def preprocesar():
    """
    Función principal que orquesta el preprocesamiento de datos.
    1. Carga los datos limpios.
    2. Crea y guarda los mapeos de etiquetas.
    3. Divide en conjuntos de entrenamiento y validación.
    4. Tokeniza los textos.
    5. Crea y guarda los objetos Dataset de PyTorch para entrenamiento y validación.
    """

    df = pd.read_csv(RAW_DATA_FILE) # Usando la constante definida arriba
    
    # 2. Preparar y guardar etiquetas
    print(f"🏷️  Generando etiquetas para la columna '{TARGET_COLUMN}'...")
    polarity2id = {polarity: idx for idx, polarity in enumerate(df[TARGET_COLUMN].unique())}
    id2polarity = {idx: polarity for polarity, idx in polarity2id.items()}
    df['label'] = df[TARGET_COLUMN].map(polarity2id)

    # Crear el directorio de salida si no existe
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar mapeo para usarlo en inferencia o análisis posterior
    path_mapeo = PROCESSED_DATA_DIR / "id2polarity.json"
    with open(path_mapeo, 'w', encoding='utf-8') as f:
        json.dump(id2polarity, f, ensure_ascii=False, indent=4)
    print(f"✅ Mapeo de etiquetas guardado en: {path_mapeo}")

    # 3. Dividir en train/validation
    print("🔪 Dividiendo datos en entrenamiento (80%) y validación (20%)...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[PREDICTOR].astype(str).tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"] # Estratificar para mantener la proporción de clases
    )

    # 4. Tokenizar
    print(f"🤖 Cargando tokenizador: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("...Tokenizando textos de entrenamiento...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    print("...Tokenizando textos de validación...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # 5. Crear y guardar Datasets
    print("📦 Creando Datasets de PyTorch...")
    train_dataset = RMDataset(train_encodings, train_labels)
    val_dataset = RMDataset(val_encodings, val_labels)

    path_train_set = PROCESSED_DATA_DIR / "train_dataset.pt"
    path_val_set = PROCESSED_DATA_DIR / "val_dataset.pt"

    torch.save(train_dataset, path_train_set)
    torch.save(val_dataset, path_val_set)
    
    print("\n🎉 ¡Preprocesamiento completado exitosamente!")
    print(f"   - Dataset de entrenamiento guardado en: {path_train_set}")
    print(f"   - Dataset de validación guardado en: {path_val_set}")


if __name__ == "__main__":
    preprocesar()