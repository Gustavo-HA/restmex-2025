from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

# Configuración de directorios y archivos
RAW_DATA_DIR = SRC_DIR / Path("../data/raw/")
INTERIM_DATA_DIR = SRC_DIR / Path("../data/interim/")
PREPROCESSED_DATA_DIR = SRC_DIR / Path("../data/preprocessed/")

RAW_DATA_PATH = RAW_DATA_DIR / "restmex-corpus.csv"
INTERIM_DATA_PATH = INTERIM_DATA_DIR / "restmex-corpus-interim.csv"

TRAIN_SET = PREPROCESSED_DATA_DIR / "train_set.csv"
TEST_SET = PREPROCESSED_DATA_DIR / "test_set.csv"


TYPE_PREPROCESSED_DIR = PREPROCESSED_DATA_DIR / "type/"
TOWN_PREPROCESSED_DIR = PREPROCESSED_DATA_DIR / "town/"
POLARITY_PREPROCESSED_DIR = PREPROCESSED_DATA_DIR / "polarity/"

# Targets
TARGET1 = "Polarity"
TARGET2 = "Town"
TARGET3 = "Type"
TARGETS = [TARGET1, TARGET2, TARGET3]


# Columnas texto
TEXT_COLUMNS = ["Title", "Review"]

# Columna predictora
PREDICTOR = "Texto_Limpio"