import torch
import numpy as np
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.nn import CrossEntropyLoss
from .dataset import RMDataset

from ..config import (
    PREPROCESSED_DATA_DIR,
    MODELS_DIR
)

# --- 2. Construcción de Rutas a partir de la Configuración ---
# Usamos las variables importadas para definir dónde leer y dónde escribir.
INPUT_DATA_PATH = PREPROCESSED_DATA_DIR / "polarity"
OUTPUT_MODEL_PATH = MODELS_DIR / "polarity_classifier"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

def entrenar():
    """
    Función principal para entrenar el modelo.
    Carga artefactos preprocesados, configura y ejecuta el Trainer de Hugging Face.
    """
    print("Iniciando el entrenamiento del modelo...")

    # Cargar artefactos preprocesados desde la ruta de entrada
    print(f"Cargando datasets desde: {INPUT_DATA_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Usando dispositivo: {device}")

    try:
        train_dataset = torch.load(INPUT_DATA_PATH / "train_dataset.pt", weights_only=False)
        val_dataset = torch.load(INPUT_DATA_PATH / "val_dataset.pt", weights_only=False)
        with open(INPUT_DATA_PATH / "id2polarity.json", 'r', encoding='utf-8') as f:
            id2polarity = json.load(f)
        num_labels = len(id2polarity)
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado. Asegúrate de haber ejecutado preprocesamiento.py primero.")
        print(e)
        return

    print("✅ Datasets cargados correctamente.")

    # Calcular pesos de clase para el desbalanceo
    print("Calculando pesos de clase para el desbalanceo...")
    train_labels = train_dataset.labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Pesos calculados.")

    # Definir función de métricas
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    # Definir Trainer personalizado que usa los pesos
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Cargar el modelo pre-entrenado desde la config
    print(f"Cargando modelo base: {MODEL_NAME} con {num_labels} etiquetas...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
    # Asegurarse de que el directorio de salida exista
    OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_PATH / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_MODEL_PATH / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none"
    )

    # Instanciar y ejecutar el Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n¡Todo listo! Iniciando entrenamiento...\n")
    trainer.train()
    print("\nEntrenamiento finalizado.")

    # Guardar el mejor modelo y el tokenizador en una carpeta final
    final_model_path = OUTPUT_MODEL_PATH / "final_model_polarity"
    trainer.save_model(final_model_path)
    print(f"🏆 Mejor modelo guardado en: {final_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(final_model_path)
    print(f"   - Tokenizador guardado junto con el modelo para facilitar la inferencia.")


if __name__ == "__main__":
    entrenar()