import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import config # Import config file
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


class ModelEvaluator:
    """Handles model evaluation on the test set."""

    def __init__(self, model, test_loader, criterion, device, is_bert=False):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion # Can be None if only metrics are needed
        self.device = device
        self.is_bert = is_bert

    def evaluate(self):
        """Evaluates the model on the test set and returns metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                if self.is_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device).unsqueeze(1)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    outputs = self.model(sequences)

                if self.criterion:
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                # Get predictions
                preds = torch.sigmoid(outputs) >= config.PREDICTION_THRESHOLD
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Flatten lists
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten().astype(int) # Ensure labels are int for metrics

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary') # Use 'binary' for positive class

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(self.test_loader) if self.criterion else None
        }

        print("\n--- Test Set Evaluation ---")
        if metrics['loss'] is not None:
            print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")

        print("\nClassification Report:")
        target_names = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL] # Assuming 0 is negative, 1 is positive
        print(classification_report(all_labels, all_preds, target_names=target_names))

        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds, target_names)


        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plots and saves the confusion matrix locally and logs to W&B."""
        if not config.LOG_WANDB and not os.path.exists(config.PLOTS_DIR):
             print("Confusion matrix plotting skipped (LOG_WANDB is False and PLOTS_DIR doesn't exist).")
             return

        cm = confusion_matrix(y_true, y_pred)
        # Create figure explicitly
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Log plot to W&B (if enabled)
        if config.LOG_WANDB and wandb.run:
            try:
                 wandb.log({"plots/confusion_matrix": wandb.Image(fig)}, commit=False)
                 print("Confusion matrix logged to W&B.")
            except Exception as e:
                print(f"Warning: Could not log confusion matrix to W&B: {e}")


        # Save the plot locally
        save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        plt.close(fig) # Close the figure


def compare_models(results_dict, metric='f1'):
    """Plots a comparison of models, saves locally and logs to W&B."""
    if not results_dict:
        print("No results to compare.")
        return

    if not config.LOG_WANDB and not os.path.exists(config.PLOTS_DIR):
        print("Model comparison plotting skipped (LOG_WANDB is False and PLOTS_DIR doesn't exist).")
        return

    models = list(results_dict.keys())
    scores = [results_dict[model].get(metric, 0) for model in models]

    df = pd.DataFrame({'Model': models, metric.capitalize(): scores})
    df = df.sort_values(by=metric.capitalize(), ascending=False)

    # Create figure explicitly
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=metric.capitalize(), y='Model', data=df, palette='viridis')
    plt.title(f'Model Comparison based on {metric.capitalize()}')
    plt.xlabel(f'{metric.capitalize()} Score')
    plt.ylabel('Model')
    plt.tight_layout()

    # Log plot to W&B (if enabled)
    if config.LOG_WANDB and wandb.run:
        try:
            wandb.log({f"plots/model_comparison_{metric}": wandb.Image(fig)}, commit=False)
            print(f"Model comparison plot ({metric}) logged to W&B.")
        except Exception as e:
            print(f"Warning: Could not log model comparison plot to W&B: {e}")


    # Save the plot locally
    save_path = os.path.join(config.PLOTS_DIR, f"model_comparison_{metric}.png")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Model comparison plot saved to {save_path}")
    plt.close(fig) # Close the figure