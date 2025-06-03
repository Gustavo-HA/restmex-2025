from ..config import INTERIM_DATA_DIR, PREDICTOR, TARGETS, PREPROCESSED_DATA_DIR
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import set_config
import numpy as np

set_config(transform_output="pandas") 

class TfidfTextVectorizer(BaseEstimator, TransformerMixin):
    """
    Tfidf compatible con sklearn.
    Utiliza la columna PREDICTOR del DataFrame de entrada.
    """
    def __init__(self, max_features: int = None, min_df: int = 1, max_df: float = 1.0, ngram_range: tuple = (1, 1)):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.is_fitted_ = False
        self.feature_names_out_ = None # Not needed if outputting NumPy

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the TfidfVectorizer on the PREDICTOR column of the input DataFrame.
        """
        if PREDICTOR not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{PREDICTOR}' column.")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range
        )
        
        self.tfidf_vectorizer.fit(X[PREDICTOR].astype(str))
        self.is_fitted_ = True
        self.feature_names_out_ = self.tfidf_vectorizer.get_feature_names_out() # Not needed
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: # Change return type hint to pd.DataFrame
        """
        Transforms the PREDICTOR column using the fitted TfidfVectorizer.
        Returns a DataFrame with TF-IDF features and the target variable.
        """
        if not self.is_fitted_:
            raise RuntimeError("TfidfTextVectorizer not fitted. Call fit() first.")
        
        if PREDICTOR not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{PREDICTOR}' column.")
        
        # Ensure the target column exists in the input X
        target_column_name = TARGETS[2] # Assuming TARGETS[2] is the 'Type' target
        if target_column_name not in X.columns:
            raise ValueError(f"Input DataFrame must contain the target column '{target_column_name}'.")

        tfidf_matrix = self.tfidf_vectorizer.transform(X[PREDICTOR].astype(str))
        
        # Convert sparse matrix to dense DataFrame with proper column names and index
        df_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.feature_names_out_,
            index=X.index # Preserve original index
        )
        
        # Add the target variable to the feature DataFrame
        df_features[target_column_name] = X[target_column_name]
        
        return df_features


class Word2VecTextVectorizer(BaseEstimator, TransformerMixin):
    """
    W2V compatible con sklearn.
    Outputting a pandas DataFrame.
    """
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 5, workers: int = 4, epochs: int = 10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.w2v_model = None
        self.is_fitted_ = False
        self.feature_names_out_ = None # Re-enable for pandas output

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the Word2Vec model on the PREDICTOR column of the input DataFrame.
        """
        if PREDICTOR not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{PREDICTOR}' column.")
        
        sentences = [str(doc).split() for doc in X[PREDICTOR] if pd.notna(doc)]
        
        if not sentences:
            raise ValueError("No valid sentences found for Word2Vec fitting. Check your input data.")

        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        self.w2v_model.build_vocab(sentences, update=False)
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.epochs)
        
        self.is_fitted_ = True
        self.feature_names_out_ = [f"w2v_{i}" for i in range(self.vector_size)] # Generate feature names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: # Change return type hint to pd.DataFrame
        """
        Transforms the PREDICTOR column into document vectors by averaging word vectors.
        Returns a DataFrame with Word2Vec features and the target variable.
        """
        if not self.is_fitted_:
            raise RuntimeError("Word2VecTextVectorizer not fitted. Call fit() first.")
            
        if PREDICTOR not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{PREDICTOR}' column.")

        target_column_name = TARGETS[2] # Assuming TARGETS[2] is the 'Type' target
        if target_column_name not in X.columns:
            raise ValueError(f"Input DataFrame must contain the target column '{target_column_name}'.")

        doc_vectors = []
        for doc in X[PREDICTOR]:
            if not isinstance(doc, str) or not doc.strip():
                doc_vectors.append(np.zeros(self.vector_size))
                continue
            
            words = str(doc).split()
            word_vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            
            if word_vectors:
                doc_vectors.append(np.mean(word_vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))
        
        # Convert to DataFrame with proper column names and index
        df_features = pd.DataFrame(
            doc_vectors,
            columns=self.feature_names_out_,
            index=X.index # Preserve original index
        )
        
        # Add the target variable to the feature DataFrame
        df_features[target_column_name] = X[target_column_name]
        
        return df_features

def get_word2vec_pipeline() -> Pipeline:
    return Pipeline([
        ('word2vec_vectorizer', Word2VecTextVectorizer(vector_size=150, window=3, min_count=3, workers=4, epochs=20))
    ])

def get_tfidf_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf_vectorizer', TfidfTextVectorizer(max_features=5000, ngram_range=(1,2)))
    ])

def save_vectorizer(vectorizer, filename: str):
    """
    Saves the vectorizer to a file using joblib.
    """

    output_dir = PREPROCESSED_DATA_DIR / "type" # Assuming 'type' is the subdir for this target
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    joblib.dump(vectorizer, filepath)
    print(f"Vectorizer saved to {filepath}")
    return filepath

def save_preprocessed_data(data: pd.DataFrame, filename: str): # Change type hint to pd.DataFrame
    """
    Saves the preprocessed data (DataFrame) to a CSV file.
    """
    # Ensure the target-specific directory exists
    output_dir = PREPROCESSED_DATA_DIR / "type" # Assuming 'type' is the subdir for this target
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    data.to_csv(filepath, index=False) # Save as CSV, no index
    print(f"Preprocessed data saved to {filepath}")
    return filepath

def preprocess():
    """
    Preprocesses the data and saves the vectorizers.
    """
    
    # Load interim data for the 'Type' target
    train_set_path = INTERIM_DATA_DIR / f"train_set_{TARGETS[2]}.csv"
    test_set_path = INTERIM_DATA_DIR / f"test_set_{TARGETS[2]}.csv"
    
    train_df = pd.read_csv(train_set_path)
    test_df = pd.read_csv(test_set_path)
    
    # Word2Vec Vectorizer
    print("\n--- Processing with Word2Vec ---")
    w2v_vectorizer = get_word2vec_pipeline()
    w2v_vectorizer.fit(train_df) # Fit on training data
    
    # Save the fitted vectorizer
    save_vectorizer(w2v_vectorizer, "word2vec_vectorizer.pkl")
    
    # Transform both train and test sets, including the target
    w2v_train_df = w2v_vectorizer.transform(train_df)
    w2v_test_df = w2v_vectorizer.transform(test_df)
    
    # Save the transformed DataFrames
    save_preprocessed_data(w2v_train_df, "w2v_train.csv")
    save_preprocessed_data(w2v_test_df, "w2v_test.csv")
    
    # TF-IDF Vectorizer
    print("\n--- Processing with TF-IDF ---")
    tfidf_vectorizer = get_tfidf_pipeline()
    tfidf_vectorizer.fit(train_df) # Fit on training data
    
    # Save the fitted vectorizer
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    
    # Transform both train and test sets, including the target
    tfidf_train_df = tfidf_vectorizer.transform(train_df)
    tfidf_test_df = tfidf_vectorizer.transform(test_df)
    
    # Save the transformed DataFrames
    save_preprocessed_data(tfidf_train_df, "tfidf_train.csv")
    save_preprocessed_data(tfidf_test_df, "tfidf_test.csv")

if __name__ == "__main__":
    preprocess()
    print("Preprocessing completed successfully.")
