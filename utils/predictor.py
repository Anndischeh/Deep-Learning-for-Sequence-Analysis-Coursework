import torch
import config
from preprocessing.text_processor import TextPreprocessor # For non-BERT models
from transformers import DistilBertTokenizer # For BERT models
from utils.dataset import pad_sequences_custom # Import the custom padding function

class SentimentPredictor:
    """Handles sentiment prediction for new text inputs."""

    def __init__(self, model, device, model_type, text_preprocessor=None, tokenizer=None, max_len=config.MAX_LEN):
        self.model = model.to(device).eval() # Ensure model is on correct device and in eval mode
        self.device = device
        self.model_type = model_type.lower() # e.g., 'lstm', 'distilbert'
        self.max_len = max_len
        self.prediction_threshold = config.PREDICTION_THRESHOLD

        if self.model_type == 'distilbert':
            if tokenizer is None:
                 raise ValueError("DistilBERT tokenizer must be provided for prediction.")
            self.tokenizer = tokenizer
            self.text_preprocessor = None # Not used for BERT prediction pipeline
        else:
            if text_preprocessor is None or text_preprocessor.word_to_index is None:
                raise ValueError("TextPreprocessor with built vocabulary must be provided for non-BERT models.")
            self.text_preprocessor = text_preprocessor
            self.tokenizer = None # Use text_preprocessor methods

    def predict(self, text):
        """Predicts the sentiment of a single text string."""
        if not isinstance(text, str):
            text = str(text) # Ensure input is a string

        self.model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            if self.model_type == 'distilbert':
                # 1. Tokenize using Hugging Face Tokenizer
                inputs = self.tokenizer(text,
                                        return_tensors="pt",
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_len)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                # 2. Model Prediction
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            else: # For RNN, CNN models
                # 1. Preprocess text using TextPreprocessor
                processed_tokens = self.text_preprocessor.preprocess_text(text) # Get list of tokens
                 # Check if tokens are empty after preprocessing
                if not processed_tokens:
                    print("Warning: Input text resulted in empty tokens after preprocessing. Predicting negative.")
                    # Handle empty sequence case: return a default prediction or raise error
                    # Option 1: Default prediction (e.g., negative)
                    return config.NEGATIVE_LABEL
                    # Option 2: Return a specific indicator
                    # return "Preprocessing resulted in empty sequence"
                    # Option 3: Use a minimal sequence if appropriate for the model
                    # sequence = [self.text_preprocessor.word_to_index.get(config.OOV_TOKEN, 0)] # e.g., just OOV

                # 2. Convert to sequence of indices
                sequence = self.text_preprocessor.text_to_sequence(processed_tokens)

                # 3. Pad sequence
                padded_sequence = pad_sequences_custom([sequence], maxlen=self.max_len, padding=config.PAD_TYPE, truncating=config.TRUNC_TYPE)[0]


                # 4. Convert to tensor
                tensor = torch.tensor(padded_sequence, dtype=torch.long).unsqueeze(0).to(self.device) # Add batch dimension

                # 5. Model Prediction
                outputs = self.model(tensor)

            # Apply sigmoid and threshold for binary classification
            probability = torch.sigmoid(outputs).squeeze().item() # Get single probability value
            prediction = 1 if probability >= self.prediction_threshold else 0

            # Map prediction to label name
            sentiment = config.POSITIVE_LABEL if prediction == 1 else config.NEGATIVE_LABEL

            return sentiment, probability # Return both label and probability