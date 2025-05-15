import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
import emoji
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import config 
import wandb



try:
    stopwords.words('english')
    word_tokenize('test')
except LookupError:
    print("NLTK data not found. Please run the download commands mentioned in README or requirements.")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """Handles text cleaning, normalization, and vocabulary building."""

    def __init__(self, vocab_size=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vocab = None
        self.word_to_index = None
        self.index_to_word = None
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.max_len = config.MAX_LEN # Store max_len for sequence conversion
        tqdm.pandas() # Enable progress bar for pandas apply

    def _lowercase(self, text):
        return text.lower()

    def _remove_html(self, text):
        return BeautifulSoup(text, "html.parser").get_text()

    def _remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def _remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    # Optional: Spelling correction (can be slow)
    def _correct_spelling(self, text):
        # Use with caution on large datasets
        # return str(TextBlob(text).correct())
        return text # Skip for now for speed

    def _remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def _handle_emojis(self, text):
        # Convert emojis to text representation (e.g., :thumbs_up:)
        # return emoji.demojize(text)
        # Or remove them
        return emoji.replace_emoji(text, replace='')


    def _tokenize(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            return []
        return word_tokenize(text)

    def _stem(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess_text(self, text, apply_stemming=True):
        """Applies the full preprocessing pipeline to a single text string."""
        if not isinstance(text, str): # Handle potential non-string data
            text = str(text)
        text = self._lowercase(text)
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._handle_emojis(text)
        text = self._remove_punctuation(text)
        # text = self._correct_spelling(text) # Optional and potentially slow
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = [token for token in tokens if token.lower() != 'br']
        if apply_stemming:
            tokens = self._stem(tokens)
        return tokens # Return list of tokens

    def preprocess_series(self, series: pd.Series, apply_stemming=True) -> pd.Series:
        """Applies preprocessing to a pandas Series."""
        print("Preprocessing text data...")
        # Use progress_apply for visual feedback
        return series.progress_apply(lambda x: self.preprocess_text(x, apply_stemming))

    def build_vocab(self, all_tokens_list):
        """Builds the vocabulary from a list of token lists."""
        print("Building vocabulary...")
        all_tokens = [token for sublist in all_tokens_list for token in sublist]

        # Count word frequencies
        word_counts = Counter(all_tokens)

        # Keep the most common words up to vocab_size - 1 (for OOV token)
        most_common_words = word_counts.most_common(self.vocab_size - 1) # Reserve one spot for OOV

        # Create word_to_index and index_to_word mappings
        self.word_to_index = {self.oov_token: 0} # Start with OOV token
        self.index_to_word = {0: self.oov_token}

        for index, (word, _) in enumerate(most_common_words, 1): # Start indexing from 1
             if word not in self.word_to_index: # Avoid duplicates if OOV happens to be common
                self.word_to_index[word] = index
                self.index_to_word[index] = word

        self.vocab = set(self.word_to_index.keys())
        print(f"Vocabulary size: {len(self.vocab)}")
        # print("Sample word_to_index:", dict(list(self.word_to_index.items())[:10]))


    def text_to_sequence(self, tokens):
        """Converts a list of tokens to a sequence of indices."""
        if self.word_to_index is None:
            raise ValueError("Vocabulary not built yet. Call build_vocab first.")
        return [self.word_to_index.get(word, self.word_to_index[self.oov_token]) for word in tokens]

    def texts_to_sequences(self, list_of_token_lists):
        """Converts a list of token lists to sequences."""
        print("Converting texts to sequences...")
        return [self.text_to_sequence(tokens) for tokens in tqdm(list_of_token_lists)]

    def visualize_word_cloud(self, df, text_col=config.TEXT_COLUMN, label_col=config.LABEL_COLUMN):
        """Generates and saves word clouds locally and logs to W&B."""
        if not config.LOG_WANDB and not os.path.exists(config.PLOTS_DIR):
             print("Word cloud generation skipped (LOG_WANDB is False and PLOTS_DIR doesn't exist).")
             return

        print("Generating word clouds...")
        try:
            # ... (logic to get positive_texts and negative_texts) ...
            positive_texts = " ".join(review for review in df[df[label_col] == config.POSITIVE_LABEL][text_col] if isinstance(review, str))
            negative_texts = " ".join(review for review in df[df[label_col] == config.NEGATIVE_LABEL][text_col] if isinstance(review, str))

            if not positive_texts and not negative_texts:
                print("Warning: No text found for generating word clouds.")
                return

            # Create figure and axes explicitly
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))

            if positive_texts:
                pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
                axs[0].imshow(pos_wordcloud, interpolation='bilinear')
                axs[0].set_title('Word Cloud - Positive Reviews')
            else:
                 axs[0].set_title('Word Cloud - Positive Reviews (No Data)')
            axs[0].axis('off')

            if negative_texts:
                neg_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(negative_texts)
                axs[1].imshow(neg_wordcloud, interpolation='bilinear')
                axs[1].set_title('Word Cloud - Negative Reviews')
            else:
                axs[1].set_title('Word Cloud - Negative Reviews (No Data)')
            axs[1].axis('off')

            plt.tight_layout()

            # Log plot to W&B (if enabled)
            if config.LOG_WANDB and wandb.run:
                try:
                    wandb.log({"plots/word_clouds": wandb.Image(fig)}, commit=False)
                    print("Word clouds logged to W&B.")
                except Exception as e:
                     print(f"Warning: Could not log word clouds to W&B: {e}")


            # Save the plot locally
            save_path = os.path.join(config.PLOTS_DIR, "word_clouds.png")
            os.makedirs(config.PLOTS_DIR, exist_ok=True)
            plt.savefig(save_path)
            print(f"Word clouds saved to {save_path}")
            plt.close(fig) # Close the figure

        except Exception as e:
            print(f"Could not generate word clouds: {e}")
            # Close figure if error occurred after creation
            if 'fig' in locals():
                plt.close(fig)

