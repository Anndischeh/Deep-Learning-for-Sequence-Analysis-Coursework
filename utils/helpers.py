# utils/helpers.py
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import config
import pandas as pd 
import seaborn as sns 
import wandb 


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def save_model(model, filepath):
    """Saves the model's state dictionary."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    # print(f"Model state_dict saved to {filepath}")

def load_model_state(model, filepath, device):
    """Loads the model's state dictionary."""
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"Model file not found at {filepath}")
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model state_dict loaded from {filepath} and moved to {device}")
    return model



def plot_training_history(history, model_name="Model"):
    """Plots training and validation loss and accuracy, saves locally and logs to W&B."""
    if not config.LOG_WANDB and not os.path.exists(config.PLOTS_DIR):
        print("Plot saving skipped (LOG_WANDB is False and PLOTS_DIR doesn't exist).")
        return # Skip if not logging and not saving locally

    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure and axes explicitly to easily pass to wandb.Image
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axs[0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axs[0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axs[0].set_title(f'{model_name} - Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    axs[1].plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    axs[1].set_title(f'{model_name} - Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    # Log plot to W&B (if enabled)
    if config.LOG_WANDB and wandb.run:
        try:
            # Use a key that reflects the plot content and model
            wandb.log({f"plots/{model_name}_training_history": wandb.Image(fig)}, commit=False) # commit=False if logging other metrics in the same step
            print("Training history plot logged to W&B.")
        except Exception as e:
            print(f"Warning: Could not log training history plot to W&B: {e}")


    # Save the plot locally
    save_path = os.path.join(config.PLOTS_DIR, f"{model_name}_training_history.png")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close(fig) # Close the figure to free memory


def load_and_explore_data(filepath=config.DATA_PATH):
    """Loads data, prints info, checks nulls, visualizes class distribution (local save + W&B log)."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return None

    print("\n--- Data Exploration ---")
    # ... (print info, head, nulls) ...

    # Check label distribution
    if config.LABEL_COLUMN in df.columns:
        print("\nLabel Distribution:")
        print(df[config.LABEL_COLUMN].value_counts())

        # Visualize label distribution
        fig = plt.figure(figsize=(6, 4)) # Capture figure
        sns.countplot(x=config.LABEL_COLUMN, data=df, palette='viridis')
        plt.title('Class Distribution (Sentiment)')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()

        # Log plot to W&B (if enabled and run active)
        if config.LOG_WANDB and wandb.run:
            try:
                wandb.log({"plots/class_distribution": wandb.Image(fig)}, commit=False)
                print("Class distribution plot logged to W&B.")
            except Exception as e:
                print(f"Warning: Could not log class distribution plot to W&B: {e}")


        # Save the plot locally
        save_path = os.path.join(config.PLOTS_DIR, "class_distribution.png")
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
        plt.close(fig) # Close the figure
    else:
        print(f"\nWarning: Label column '{config.LABEL_COLUMN}' not found in the dataset.")

    print("------------------------\n")
    return df

