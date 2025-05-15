# File Paths
DATA_PATH = "data/IMDB Dataset.csv"
MODEL_SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"

# Data Parameters
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
POSITIVE_LABEL = 'positive' # Label value for positive sentiment
NEGATIVE_LABEL = 'negative' # Label value for negative sentiment
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15 # Validation split from the *training* set

# Preprocessing Parameters
MAX_LEN = 256 # Max sequence length after tokenization
VOCAB_SIZE = 10000 # Max vocabulary size (for non-transformer models)
TRUNC_TYPE = 'post'
PAD_TYPE = 'post'
OOV_TOKEN = "<OOV>" # Out-of-vocabulary token

# Model Hyperparameters (Examples - Tune these!)
EMBEDDING_DIM = 128
HIDDEN_DIM_RNN = 128
HIDDEN_DIM_GRU = 128
HIDDEN_DIM_LSTM = 128
N_LAYERS_RNN = 2
N_LAYERS_GRU = 2
N_LAYERS_LSTM = 2
BIDIRECTIONAL_LSTM = True
DROPOUT_RATE = 0.5
OUTPUT_DIM = 1 # Single output for binary classification (using BCEWithLogitsLoss)

# CNN Specific
N_FILTERS_CNN = 100
FILTER_SIZES_CNN = [3, 4, 5]

# Transformer Specific (DistilBERT)
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'

# Training Parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 10 # Start with a small number for testing
DEVICE = 'cuda' # or 'cpu'

# Evaluation
METRICS = ['accuracy', 'precision', 'recall', 'f1']

# Prediction
PREDICTION_THRESHOLD = 0.5

# --- ADD OR VERIFY THIS SECTION ---
# W&B Configuration (Optional - can be set via environment variables or wandb login)
WANDB_PROJECT = "Deep Learning for Sequential Analysis" # Your W&B project name
WANDB_ENTITY = None  # Replace with your W&B username or team name if needed
WANDB_MODE = "online" # "online", "offline", "disabled"
LOG_WANDB = True # Flag to enable/disable W&B logging globally