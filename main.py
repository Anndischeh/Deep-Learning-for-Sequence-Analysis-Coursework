import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer
import config
import wandb
from preprocessing.text_processor import TextPreprocessor
from utils.dataset import create_dataloaders
from models.rnn_models import RNNClassifier, LSTMClassifier, GRUClassifier
from models.cnn_model import CNNClassifier, DCNNClassifier
from models.transformer_model import DistilBERTClassifier
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator # Removed compare_models if not used here
from utils.helpers import set_seed, load_model_state, plot_training_history, load_and_explore_data
from utils.predictor import SentimentPredictor


# Define available models
AVAILABLE_MODELS = {
    'rnn': RNNClassifier,
    'lstm': LSTMClassifier,
    'bilstm': LSTMClassifier, # Will set bidirectional=True later
    'gru': GRUClassifier,
    'cnn': CNNClassifier,
    'dcnn': DCNNClassifier,
    'distilbert': DistilBERTClassifier
}


def main(args):
    # --- Basic Setup ---
    set_seed()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # --- W&B Initialization ---
    run = None # Initialize run object
    if config.LOG_WANDB:
        try:
            # Define base config dictionary
            wandb_config = {
                "model_type": args.model_type,
                "mode": args.mode,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "n_epochs": config.N_EPOCHS,
                "max_len": config.MAX_LEN,
                "dropout": config.DROPOUT_RATE,
                "optimizer": "AdamW" if args.model_type == 'distilbert' else "Adam",
                "device": str(device),
                "test_split_size": config.TEST_SIZE,
                "validation_split_size": config.VALIDATION_SIZE,
                "prediction_threshold": config.PREDICTION_THRESHOLD,
            }
            # Add model-specific hyperparameters conditionally
            if args.model_type != 'distilbert':
                 wandb_config["vocab_size_limit"] = config.VOCAB_SIZE # The limit set
                 wandb_config["embedding_dim"] = config.EMBEDDING_DIM
                 wandb_config["oov_token"] = config.OOV_TOKEN
                 wandb_config["pad_type"] = config.PAD_TYPE
                 wandb_config["trunc_type"] = config.TRUNC_TYPE

                 if args.model_type == 'rnn':
                     wandb_config["hidden_dim_rnn"] = config.HIDDEN_DIM_RNN
                     wandb_config["n_layers_rnn"] = config.N_LAYERS_RNN

                 elif args.model_type in ['lstm', 'bilstm']:
                     wandb_config["hidden_dim_lstm"] = config.HIDDEN_DIM_LSTM
                     wandb_config["n_layers_lstm"] = config.N_LAYERS_LSTM
                     wandb_config["bidirectional_lstm"] = True if args.model_type == 'bilstm' else config.BIDIRECTIONAL_LSTM
                     
                 elif args.model_type == 'gru':
                     wandb_config["hidden_dim_gru"] = config.HIDDEN_DIM_GRU
                     wandb_config["n_layers_gru"] = config.N_LAYERS_GRU
                     # Assuming GRU uses BIDIRECTIONAL_LSTM config for consistency here, adjust if needed
                     wandb_config["bidirectional_gru"] = config.BIDIRECTIONAL_LSTM

                 elif args.model_type == 'cnn':
                      wandb_config["n_filters_cnn"] = config.N_FILTERS_CNN
                      wandb_config["filter_sizes_cnn"] = config.FILTER_SIZES_CNN

                 elif args.model_type == 'dcnn':
                      wandb_config["n_filters_cnn"] = config.N_FILTERS_CNN
                      wandb_config["filter_sizes_cnn"] = config.FILTER_SIZES_CNN

            else: # DistilBERT specific
                 wandb_config["distilbert_model_name"] = config.DISTILBERT_MODEL_NAME


            run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                config=wandb_config, # Log the assembled config
                name=f"{args.model_type}_{args.mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}", # Example run name
                mode=config.WANDB_MODE,
                reinit=True, # Allows multiple runs in one script if needed, usually fine here
                save_code=True, # Save main script to W&B
            )
            print(f"W&B Run initialized: {wandb.run.get_url()}")
        except Exception as e:
             print(f"Error initializing W&B: {e}. Disabling W&B logging for this run.")
             config.LOG_WANDB = False # Disable logging if init fails


    try: # Wrap main logic in try...finally to ensure wandb.finish() runs
        # --- Data Loading and Initial Exploration ---
        # Plots generated here will be logged to W&B by the modified function
        df = load_and_explore_data(config.DATA_PATH)
        if df is None:
            return

        # Optional: Sample data for faster dev/testing
        # sample_size = 5000
        # df = df.sample(n=sample_size, random_state=42)
        # print(f"--- Using sampled data: {len(df)} rows ---")
        # if config.LOG_WANDB and run:
        #    wandb.config.update({"using_sampled_data": True, "sample_size": sample_size})


        # --- Preprocessing ---
        # Word clouds generated here will be logged to W&B
        vocab_size = None
        text_preprocessor = None
        bert_tokenizer = None

        if args.model_type != 'distilbert':
            print(f"\n--- Initializing Text Preprocessor for {args.model_type} ---")
            text_preprocessor = TextPreprocessor(vocab_size=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN)
            processed_tokens_list = text_preprocessor.preprocess_series(df[config.TEXT_COLUMN]).tolist()
            text_preprocessor.build_vocab(processed_tokens_list)
            vocab_size = len(text_preprocessor.vocab) # Get actual vocab size
            # Update W&B config with actual vocab size
            if config.LOG_WANDB and wandb.run:
                 wandb.config.update({"actual_vocab_size": vocab_size}, allow_val_change=True)

            sequences = text_preprocessor.texts_to_sequences(processed_tokens_list)
            text_preprocessor.visualize_word_cloud(df) # Logs to W&B if enabled
            data_input = sequences
            labels_input = df[config.LABEL_COLUMN].tolist()
            is_bert_flag = False

        else: # DistilBERT
            print("\n--- Initializing DistilBERT Tokenizer ---")
            bert_tokenizer = DistilBertTokenizer.from_pretrained(config.DISTILBERT_MODEL_NAME)
            data_input = df[config.TEXT_COLUMN] # Pass the pandas Series directly
            labels_input = df[config.LABEL_COLUMN].tolist()
            is_bert_flag = True
            # Optionally generate word clouds from raw text
            if config.LOG_WANDB or os.path.exists(config.PLOTS_DIR):
               TextPreprocessor().visualize_word_cloud(df) # Instantiate temporarily

        # --- Data Loaders ---
        print("\n--- Creating DataLoaders ---")
        train_loader, val_loader, test_loader = create_dataloaders(
            data=data_input,            # The sequences list or raw text Series
            labels=labels_input,        # The list of labels (e.g., 'positive'/'negative' or 0/1)
            batch_size=config.BATCH_SIZE, # The batch size from your config
            is_bert=is_bert_flag,       # The boolean flag indicating model type
            tokenizer=bert_tokenizer,   # The tokenizer object (None for non-BERT)
            test_size=config.TEST_SIZE, # Pass test_size explicitly
            val_size=config.VALIDATION_SIZE # Pass val_size explicitly
        )

        # --- Model Selection ---
        print(f"\n--- Selecting Model: {args.model_type} ---")
        if args.model_type not in AVAILABLE_MODELS:
             raise ValueError(f"Unknown model type: {args.model_type}. Choose from {list(AVAILABLE_MODELS.keys())}")

        ModelClass = AVAILABLE_MODELS[args.model_type]
        # Base parameters common to most models (or used by BERT)
        model_params = {'output_dim': config.OUTPUT_DIM, 'dropout': config.DROPOUT_RATE}

        # Instantiate model based on type, adding necessary parameters
        if args.model_type == 'distilbert':
            model = ModelClass(
                output_dim=config.OUTPUT_DIM,
                dropout=config.DROPOUT_RATE,
                model_name=config.DISTILBERT_MODEL_NAME # Pass model name
            )


        elif args.model_type == 'cnn':
            model = ModelClass(
                vocab_size=vocab_size,
                embedding_dim=config.EMBEDDING_DIM,
                n_filters=config.N_FILTERS_CNN,
                filter_sizes=config.FILTER_SIZES_CNN,
                output_dim=config.OUTPUT_DIM,
                dropout=config.DROPOUT_RATE,
                pad_idx=text_preprocessor.word_to_index.get('<pad>', 0) # Assuming pad index is 0 if not explicitly built
            )

        elif args.model_type == 'dcnn':
            model = ModelClass(
                vocab_size=vocab_size,
                embedding_dim=config.EMBEDDING_DIM,
                n_filters=config.N_FILTERS_CNN,
                filter_sizes=config.FILTER_SIZES_CNN,
                output_dim=config.OUTPUT_DIM,
                dropout=config.DROPOUT_RATE,
                pad_idx=text_preprocessor.word_to_index.get('<pad>', 0) # Assuming pad index is 0 if not explicitly built
            )
        elif args.model_type == 'rnn':
            model = ModelClass(
                vocab_size=vocab_size,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM_RNN,
                output_dim=config.OUTPUT_DIM,
                n_layers=config.N_LAYERS_RNN,
                dropout=config.DROPOUT_RATE,
                pad_idx=text_preprocessor.word_to_index.get('<pad>', 0)
            )
        elif args.model_type == 'lstm':
             model = ModelClass( # LSTMClassifier
                 vocab_size=vocab_size,
                 embedding_dim=config.EMBEDDING_DIM,
                 hidden_dim=config.HIDDEN_DIM_LSTM,
                 output_dim=config.OUTPUT_DIM,
                 n_layers=config.N_LAYERS_LSTM,
                 bidirectional=config.BIDIRECTIONAL_LSTM, # Use config setting
                 dropout=config.DROPOUT_RATE,
                 pad_idx=text_preprocessor.word_to_index.get('<pad>', 0)
             )
        elif args.model_type == 'bilstm':
             model = ModelClass( # LSTMClassifier
                 vocab_size=vocab_size,
                 embedding_dim=config.EMBEDDING_DIM,
                 hidden_dim=config.HIDDEN_DIM_LSTM,
                 output_dim=config.OUTPUT_DIM,
                 n_layers=config.N_LAYERS_LSTM,
                 bidirectional=True, # Force bidirectional for bilstm type
                 dropout=config.DROPOUT_RATE,
                 pad_idx=text_preprocessor.word_to_index.get('<pad>', 0)
             )
        elif args.model_type == 'gru':
             model = ModelClass( # GRUClassifier
                 vocab_size=vocab_size,
                 embedding_dim=config.EMBEDDING_DIM,
                 hidden_dim=config.HIDDEN_DIM_GRU,
                 output_dim=config.OUTPUT_DIM,
                 n_layers=config.N_LAYERS_GRU,
                 # Reuse BiLSTM setting for consistency, or define specific GRU bidir config
                 bidirectional=config.BIDIRECTIONAL_LSTM,
                 dropout=config.DROPOUT_RATE,
                 pad_idx=text_preprocessor.word_to_index.get('<pad>', 0)
             )
        else:
             # Fallback or raise error for unhandled but listed models
             raise NotImplementedError(f"Model instantiation logic missing for {args.model_type}")


        # --- Optimizer and Loss ---
        if args.model_type == 'distilbert':
            # AdamW is often recommended for Transformers
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        criterion = nn.BCEWithLogitsLoss().to(device) # Suitable for binary classification with single output logit
        model.to(device) # Move model to device

        print("\nModel Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")
        # Log parameter count to W&B
        if config.LOG_WANDB and wandb.run:
            wandb.config.update({"trainable_params": total_params})


        model_filename = f"{args.model_type}_sentiment_model.pt"
        model_save_path = os.path.join(config.MODEL_SAVE_DIR, model_filename)


        # --- Training ---
        if args.mode in ['train', 'all']:
            print("\n--- Starting Training Phase ---")
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_save_path=model_save_path,
                is_bert=is_bert_flag
            )
            history = trainer.train(config.N_EPOCHS) # Training logs metrics/plots to W&B internally
            # Plot training history (will also log to W&B via the helper function)
            plot_training_history(history, model_name=args.model_type.upper())
            # Load best model saved during training for subsequent evaluation/prediction
            print(f"Loading best model from {model_save_path} for evaluation...")
            # Need to ensure the model variable holds the loaded state dict
            model = load_model_state(model, model_save_path, device)

        # --- Evaluation ---
        if args.mode in ['evaluate', 'all']:
            print("\n--- Starting Evaluation Phase ---")
            # Load model only if in 'evaluate' mode and not already loaded from training
            if args.mode == 'evaluate':
                 if not os.path.exists(model_save_path):
                     print(f"Error: Model file not found at {model_save_path}. Train the model first ('train' mode).")
                     return # Exit if model not found for evaluation
                 print(f"Loading model from {model_save_path} for evaluation...")
                 # Reload the architecture before loading state dict if not continuing from training
                 # (The 'model' variable might already hold the correct architecture if mode='all')
                 model = load_model_state(model, model_save_path, device)


            evaluator = ModelEvaluator(
                model=model,
                test_loader=test_loader,
                criterion=criterion, # Pass criterion to calculate test loss
                device=device,
                is_bert=is_bert_flag
            )
            test_metrics = evaluator.evaluate() # Confusion matrix plotted/logged inside

            # Log final test metrics to W&B summary
            if config.LOG_WANDB and wandb.run and test_metrics:
                 # Prefix test metrics for clarity in W&B UI
                wandb_summary_metrics = {f"test/{k}": v for k, v in test_metrics.items() if v is not None}
                try:
                    wandb.summary.update(wandb_summary_metrics)
                    print("Test metrics logged to W&B summary.")
                except Exception as e:
                    print(f"Warning: Could not log test metrics to W&B summary: {e}")

            # Store results if comparing multiple models locally (optional)
            # Example: all_results[args.model_type] = test_metrics


        # --- Prediction ---
        if args.mode in ['predict', 'all'] and args.text:
            print("\n--- Starting Prediction Phase ---")
            # Ensure model is loaded (either from training or explicitly)
            if args.mode == 'predict': # Need to load if only predicting
                if not os.path.exists(model_save_path):
                    print(f"Error: Model file not found at {model_save_path}. Train the model first ('train' mode).")
                    return
                print(f"Loading model from {model_save_path} for prediction...")
                # Re-initialize model architecture before loading state *if not already done*
                # If mode='all', 'model' should already be loaded from training/evaluation
                # If mode='predict', we need to instantiate and load fresh
                ModelClassPred = AVAILABLE_MODELS[args.model_type]
                loaded_model_arch = None # Define outside conditional

                if args.model_type == 'distilbert':
                     loaded_model_arch = ModelClassPred(output_dim=config.OUTPUT_DIM, dropout=config.DROPOUT_RATE, model_name=config.DISTILBERT_MODEL_NAME)
                elif args.model_type == 'dcnn':
                    loaded_model_arch = ModelClassPred(vocab_size=vocab_size, embedding_dim=config.EMBEDDING_DIM, n_filters=config.N_FILTERS_CNN, filter_sizes=config.FILTER_SIZES_CNN, output_dim=config.OUTPUT_DIM, dropout=config.DROPOUT_RATE, pad_idx=text_preprocessor.word_to_index.get('<pad>', 0))
                elif args.model_type == 'rnn':
                    loaded_model_arch = ModelClassPred(vocab_size=vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_RNN, output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_RNN, dropout=config.DROPOUT_RATE, pad_idx=text_preprocessor.word_to_index.get('<pad>', 0))
                elif args.model_type == 'lstm':
                    loaded_model_arch = ModelClassPred(vocab_size=vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_LSTM, output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_LSTM, bidirectional=config.BIDIRECTIONAL_LSTM, dropout=config.DROPOUT_RATE, pad_idx=text_preprocessor.word_to_index.get('<pad>', 0))
                elif args.model_type == 'bilstm':
                     loaded_model_arch = ModelClassPred(vocab_size=vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_LSTM, output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_LSTM, bidirectional=True, dropout=config.DROPOUT_RATE, pad_idx=text_preprocessor.word_to_index.get('<pad>', 0))
                elif args.model_type == 'gru':
                     loaded_model_arch = ModelClassPred(vocab_size=vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_GRU, output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_GRU, bidirectional=config.BIDIRECTIONAL_LSTM, dropout=config.DROPOUT_RATE, pad_idx=text_preprocessor.word_to_index.get('<pad>', 0))

                if loaded_model_arch is None:
                     raise ValueError(f"Could not determine model architecture for prediction with type {args.model_type}")

                # Load the state dict into the freshly initialized architecture
                model = load_model_state(loaded_model_arch, model_save_path, device)
                # We also need the preprocessor/tokenizer for prediction
                # If mode='predict' and not 'distilbert', we need to re-run parts of preprocessing
                if args.model_type != 'distilbert' and text_preprocessor is None:
                     print("Re-initializing preprocessor for prediction...")
                     # This assumes the vocab is consistent, might be better to save/load preprocessor state
                     text_preprocessor = TextPreprocessor(vocab_size=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN)
                     # Ideally, load vocab from a file saved during training instead of rebuilding
                     print("Warning: Rebuilding vocab for prediction mode; load preprocessor state for production.")
                     temp_tokens = text_preprocessor.preprocess_series(df[config.TEXT_COLUMN]).tolist() # Inefficient
                     text_preprocessor.build_vocab(temp_tokens) # Inefficient
                     vocab_size = len(text_preprocessor.vocab)
                elif args.model_type == 'distilbert' and bert_tokenizer is None:
                     print("Re-initializing BERT tokenizer for prediction...")
                     bert_tokenizer = DistilBertTokenizer.from_pretrained(config.DISTILBERT_MODEL_NAME)


            # Instantiate predictor with the loaded model and necessary tokenizer/preprocessor
            predictor = SentimentPredictor(
                model=model,
                device=device,
                model_type=args.model_type,
                text_preprocessor=text_preprocessor, # Will be None for distilbert
                tokenizer=bert_tokenizer, # Will be None for non-bert
                max_len=config.MAX_LEN
            )

            # Predict sentiment
            sentiment, probability = predictor.predict(args.text)
            print(f"\nInput Text: '{args.text}'")
            print(f"Predicted Sentiment: {sentiment} (Probability: {probability:.4f})")


        elif args.mode == 'predict' and not args.text:
            print("Error: Please provide input text for prediction using --text 'Your sentence here'")

        # Optional: Add model comparison logic here if results across runs are stored

    except Exception as e:
         print(f"\n--- An error occurred: {e} ---")
         import traceback
         traceback.print_exc() # Print detailed traceback

    finally:
        # --- W&B Finalization ---
        if config.LOG_WANDB and run: # Check if run was initialized
             wandb.finish()
             print("W&B run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Sentiment Analysis Pipeline with W&B")
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Type of model to use.')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'predict', 'all'],
                        help='Operation mode: train, evaluate, predict, or all.')
    parser.add_argument('--text', type=str, default=None,
                        help='Input text for prediction mode.')
    # Add arguments for overriding config parameters if needed
    # parser.add_argument('--epochs', type=int, help='Override number of training epochs.')
    # parser.add_argument('--lr', type=float, help='Override learning rate.')
    # parser.add_argument('--batch_size', type=int, help='Override batch size.')

    args = parser.parse_args()

    # --- Override Configs (Example) ---
    # if args.epochs:
    #     config.N_EPOCHS = args.epochs
    # if args.lr:
    #     config.LEARNING_RATE = args.lr
    # if args.batch_size:
    #      config.BATCH_SIZE = args.batch_size

    main(args)