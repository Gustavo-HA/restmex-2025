import re
from functools import lru_cache
import joblib

import pandas as pd
import numpy as np
try:
    from nltk.corpus import stopwords
except ImportError: # Use ImportError for specific exceptions
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
from .config import RAW_DATA_PATH, INTERIM_DATA_DIR, TEXT_COLUMNS, TARGETS, PREDICTOR # Assuming these are defined in config
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import set_config 

set_config(transform_output="pandas")


class ArreglaMojibake(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para arreglar mojibake en textos.
    """
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformación para arreglar mojibake en las columnas especificadas.
        """
        X_copy = X.copy()
        for variable in self.variables:
            # Apply _arregla_mojibake to string columns, handle non-strings gracefully
            X_copy[variable] = X_copy[variable].apply(
                lambda x: self._arregla_mojibake(x) if isinstance(x, str) else x
            )
        return X_copy

    @staticmethod
    def _arregla_mojibake(texto: str) -> str:
        """
        Convierte un texto con mojibake (texto mal codificado) a texto legible.
        """
        texto = texto.lower()
        try:
            return texto.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError): # Catch specific exceptions
            return texto


@lru_cache(maxsize=None)
def get_stopwords(lang='spanish'):
    return set(stopwords.words(lang))

class QuitaStopwords(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para quitar stopwords en textos.
    """
    def __init__(self, variables, lang='spanish'):
        self.lang = lang
        self.stopwords = get_stopwords(lang)
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformación para quitar stopwords en las columnas especificadas.
        """
        X_copy = X.copy()
        for variable in self.variables:
            X_copy[variable] = X_copy[variable].apply(self._quita_stopwords)
        return X_copy

    def _quita_stopwords(self, texto: str) -> str:
        """
        Elimina las stopwords de un texto.
        """
        if not isinstance(texto, str):
            return " "
        
        texto = re.sub(r'[^a-záéíóúñü\s]', '', texto) # Keep Spanish accented characters
        palabras = texto.split()
        palabras_filtradas = [palabra for palabra in palabras if palabra not in self.stopwords]
        return ' '.join(palabras_filtradas)
    
    
class SpanishStemmer(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para aplicar stemming en textos en español.
    """
    def __init__(self, variables, lang='spanish'):
        self.variables = variables
        self.stemmer = SnowballStemmer(lang)

    def fit(self, X, y=None):
        return self

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformación para aplicar stemming en las columnas especificadas.
        """
        X_copy = X.copy()
        for variable in self.variables:
            X_copy[variable] = X_copy[variable].apply(self._stem)
        return X_copy

    def _stem(self, texto: str) -> str:
        """
        Aplica el stemming a un texto.
        """
        if not isinstance(texto, str):
            return " "
        
        palabras = texto.split()
        palabras_stemmed = [self.stemmer.stem(palabra) for palabra in palabras]
        return ' '.join(palabras_stemmed)
    
    
class DropFeatures(BaseEstimator, TransformerMixin):
    """Dropping Features Which Are Less Significant"""

    def __init__(self, variables_to_drop : list):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        # Ensure variables_to_drop are actually in the DataFrame before dropping
        actual_variables_to_drop = [col for col in self.variables_to_drop if col in X_copy.columns]
        if actual_variables_to_drop:
            X_copy = X_copy.drop(actual_variables_to_drop, axis=1)
        return X_copy
    
    
class JuntarFeatures(BaseEstimator, TransformerMixin):
    """Juntar Features de texto significativas en una sola columna"""
    def __init__(self, variables_to_join : list , new_column : str):
        self.variables_to_join = variables_to_join
        self.new_column = new_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        # Ensure all variables_to_join exist and handle potential NaNs before joining
        for var in self.variables_to_join:
            X_copy[var] = X_copy[var].astype(str).fillna("") # Convert to string and fill NaNs
        
        X_copy[self.new_column] = X_copy[self.variables_to_join].agg(" ".join, axis = 1)
        X_copy[self.new_column] = X_copy[self.new_column].str.strip().replace(r'\s+', ' ', regex=True) # Clean up extra spaces
        return X_copy
    


### pipelines

# Pipeline para el dataset de RESTMEX.
def get_pipeline_completo() -> Pipeline:
    """
    Devuelve un pipeline completo para el preprocesamiento del dataset de RESTMEX.
    """
    return Pipeline(
        [
            ('Arreglar mojibakes', ArreglaMojibake(TEXT_COLUMNS)),
            ("Minúsculas y quitar stopwords", QuitaStopwords(TEXT_COLUMNS)),
            ("Stemming", SpanishStemmer(TEXT_COLUMNS)),
            ("Guardar en una columna", JuntarFeatures(TEXT_COLUMNS, PREDICTOR)),
            ("Quitar features no deseadas", DropFeatures(TEXT_COLUMNS))
        ]
    )
    
def get_pipeline_lower() -> Pipeline:
    """
    Devuelve un pipeline para el preprocesamiento del dataset de RESTMEX sin stemming,
    ni quitar stopwords para BETO.
    """
    return Pipeline(
        [
            ('Arreglar mojibakes', ArreglaMojibake(TEXT_COLUMNS)),
            ("Guardar en una columna", JuntarFeatures(TEXT_COLUMNS, PREDICTOR)),
            ("Quitar features no deseadas", DropFeatures(TEXT_COLUMNS))
        ]
    )
    
    
### Data handling functions

def read_raw_data() -> pd.DataFrame:
    """
    Carga el dataset crudo desde el archivo CSV.
    """
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8')
    return df

def save_preprocessor(pipeline: Pipeline, filename: str) -> None:
    """
    Guarda el pipeline de preprocesamiento en un archivo.
    """
    joblib.dump(pipeline, filename)
    
def save_dataset(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda el DataFrame preprocesado en un archivo CSV.
    """
    df.to_csv(filename, index=False, encoding='utf-8')



def clean_data():
    # Load the raw data once
    df_raw = read_raw_data()
    
    # Define features (X) by dropping all target columns from the raw DataFrame
    X_features_full = df_raw.drop(columns=TARGETS)
    
    print("Starting data cleaning and splitting for multiple targets...")

    # --- For Polarity (TARGETS[0]) ---
    print(f"\nProcessing for target: {TARGETS[0]}")
    y_polarity = df_raw[TARGETS[0]]
    
    X_train_polarity, X_test_polarity, y_train_polarity, y_test_polarity = train_test_split(
        X_features_full, y_polarity, test_size=0.2, random_state=42, stratify=y_polarity
    )
    
    soft_pipeline_polarity = get_pipeline_lower()
    soft_pipeline_polarity.fit(X_train_polarity)
    
    X_train_polarity_transformed = soft_pipeline_polarity.transform(X_train_polarity)
    X_test_polarity_transformed = soft_pipeline_polarity.transform(X_test_polarity)

    df_train_polarity_final = X_train_polarity_transformed.copy()
    df_train_polarity_final[TARGETS[0]] = y_train_polarity.values
    df_test_polarity_final = X_test_polarity_transformed.copy()
    df_test_polarity_final[TARGETS[0]] = y_test_polarity.values

    save_dataset(df_train_polarity_final, INTERIM_DATA_DIR / f"train_set_{TARGETS[0]}.csv")
    save_dataset(df_test_polarity_final, INTERIM_DATA_DIR / f"test_set_{TARGETS[0]}.csv")
    save_preprocessor(soft_pipeline_polarity, INTERIM_DATA_DIR / f"soft_preprocessor_{TARGETS[0]}.pkl")
    print(f"Finished processing for target: {TARGETS[0]}")

    # --- For Town (TARGETS[1]) ---
    print(f"\nProcessing for target: {TARGETS[1]}")
    y_town = df_raw[TARGETS[1]]
    
    X_train_town, X_test_town, y_train_town, y_test_town = train_test_split(
        X_features_full, y_town, test_size=0.2, random_state=42, stratify=y_town
    )
    
    soft_pipeline_town = get_pipeline_lower()
    soft_pipeline_town.fit(X_train_town)
    
    X_train_town_transformed = soft_pipeline_town.transform(X_train_town)
    X_test_town_transformed = soft_pipeline_town.transform(X_test_town)

    df_train_town_final = X_train_town_transformed.copy()
    df_train_town_final[TARGETS[1]] = y_train_town.values
    df_test_town_final = X_test_town_transformed.copy()
    df_test_town_final[TARGETS[1]] = y_test_town.values

    save_dataset(df_train_town_final, INTERIM_DATA_DIR / f"train_set_{TARGETS[1]}.csv")
    save_dataset(df_test_town_final, INTERIM_DATA_DIR / f"test_set_{TARGETS[1]}.csv")
    save_preprocessor(soft_pipeline_town, INTERIM_DATA_DIR / f"soft_preprocessor_{TARGETS[1]}.pkl")
    print(f"Finished processing for target: {TARGETS[1]}")

    # --- For Type (TARGETS[2]) ---
    print(f"\nProcessing for target: {TARGETS[2]}")
    y_type = df_raw[TARGETS[2]]
    
    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
        X_features_full, y_type, test_size=0.2, random_state=42, stratify=y_type
    )
    
    complete_pipeline_type = get_pipeline_completo() 
    complete_pipeline_type.fit(X_train_type)
    
    X_train_type_transformed = complete_pipeline_type.transform(X_train_type)
    X_test_type_transformed = complete_pipeline_type.transform(X_test_type)

    df_train_type_final = X_train_type_transformed.copy()
    df_train_type_final[TARGETS[2]] = y_train_type.values
    df_test_type_final = X_test_type_transformed.copy()
    df_test_type_final[TARGETS[2]] = y_test_type.values

    save_dataset(df_train_type_final, INTERIM_DATA_DIR / f"train_set_{TARGETS[2]}.csv")
    save_dataset(df_test_type_final, INTERIM_DATA_DIR / f"test_set_{TARGETS[2]}.csv")
    save_preprocessor(complete_pipeline_type, INTERIM_DATA_DIR / f"complete_preprocessor_{TARGETS[2]}.pkl")
    print(f"Finished processing for target: {TARGETS[2]}")
    
    print("\nAll preprocessing completed and datasets saved for each target.")



if __name__ == "__main__":
    clean_data()
    