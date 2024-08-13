import re
import string
import nltk
from collections import Counter
from typing import List, Dict, Tuple
from nltk.corpus import wordnet
import unicodedata
import numpy as np
# Download NLTK resources
nltk.download('wordnet')

class BambaraTokenizer:
    def __init__(self, vocab: Dict[str, int] = None, special_tokens: Dict[str, str] = None, stopwords: List[str] = None):
        """
        Initialize the tokenizer with optional special tokens and stopwords.
        :param special_tokens: Dictionary of special tokens like {'<UNK>': 'unknown', '<PAD>': 'padding'}
        :param stopwords: List of stopwords to be removed from the text
        """
        self.vocab = vocab if vocab else {}
        self.special_characters = ['ɛ', 'ɔ', 'Ɛ', 'Ɔ', 'Ŋ', 'ɲ']
        self.special_tokens = special_tokens if special_tokens else {}
        self.stopwords = set(stopwords) if stopwords else set()
        self.token_to_id = {}
        self.id_to_token = {}



    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by handling special characters and normalizing it.
        :param text: Input Bambara text
        :return: Preprocessed text
        """
        text = text.lower()  # Convert to lowercase
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the Bambara text while preserving special characters.
        :param text: Input Bambara text
        :return: List of tokens
        """
        text = self.preprocess_text(text)
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens

    def remove_stopwords(self, tokens: List[str], stopwords: List[str]= None) -> List[str]:
        """
        Remove stopwords from the token list.
        :param tokens: List of tokens
        :return: List of tokens with stopwords removed
        """
        self.stopwords = set(stopwords) if stopwords else set(string.punctuation)
        return [token for token in tokens if token not in self.stopwords]

    def tokens_to_text(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back into a string.
        :param tokens: List of tokens
        :return: Joined string from tokens
        """
        return ' '.join(tokens)

    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """
        Replace tokens with special tokens if they exist.
        :param tokens: List of tokens
        :return: List of tokens with special tokens applied
        """
        return [self.special_tokens.get(token, token) for token in tokens]

    def get_token_frequencies(self, tokens: List[str]) -> Dict[str, int]:
        """
        Get the frequency of each token in the list.
        :param tokens: List of tokens
        :return: Dictionary with token frequencies
        """
        return dict(Counter(tokens))

    def get_unique_tokens(self, tokens: List[str]) -> List[str]:
        """
        Get a list of unique tokens from the token list.
        :param tokens: List of tokens
        :return: List of unique tokens
        """
        return list(set(tokens))

    def sort_tokens_by_frequency(self, tokens: List[str], descending: bool = True) -> List[tuple[str, int]]:
        """
        Sort tokens by frequency.
        :param tokens: List of tokens
        :param descending: Whether to sort in descending order
        :return: List of tuples with tokens and their frequencies, sorted by frequency
        """
        token_freq = self.get_token_frequencies(tokens)
        return sorted(token_freq.items(), key=lambda item: item[1], reverse=descending)

    def ngram(self, tokens: List[str], n: int) -> List[tuple[str, ...]]:
        """
        Generate n-grams from the list of tokens.
        :param tokens: List of tokens
        :param n: The number of elements in each n-gram
        :return: List of n-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def pad_sequences(self, tokens: List[str], max_len: int, padding: str = '<PAD>') -> List[str]:
        """
        Pad the sequence of tokens to a maximum length.
        :param tokens: List of tokens
        :param max_len: The maximum length of the sequence
        :param padding: The padding token to use
        :return: List of tokens padded to max_len
        """
        if len(tokens) < max_len:
            tokens.extend([padding] * (max_len - len(tokens)))
        return tokens[:max_len]

    def tokenize_numbers(self, tokens: List[str]) -> List[str]:
        """
        Handle numbers in the token list.
        :param tokens: List of tokens
        :return: List of tokens with numbers handled
        """
        return ['<NUM>' if token.isdigit() else token for token in tokens]

    def character_tokenization(self, text: str) -> List[str]:
        """
        Tokenize the text at the character level.
        :param text: Input Bambara text
        :return: List of characters
        """
        return list(text)

    def detect_unknown_words(self, tokens: List[str], known_words: List[str]) -> List[str]:
        """
        Detect and return tokens that are not in the list of known words.
        :param tokens: List of tokens
        :param known_words: List of known words
        :return: List of unknown words
        """
        return [token for token in tokens if token not in known_words]

    def build_vocab(self, tokens: List[str]):
        """
        Build a vocabulary from a list of tokens.
        :param tokens: List of tokens
        :return: None
        """
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1  # start indexing from 1

    def tokenize_with_vocab(self, text: str) -> List[int]:
        """
        Tokenize text and return token IDs based on the vocabulary.
        :param text: Input text
        :return: List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get('<UNK>')) for token in tokens]

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        :return: Size of the vocabulary
        """
        return len(self.vocab)

    # Add BPE-related methods to the class

    def bpe_tokenize(self, text: str) -> List[str]:
        """
        Apply Byte-Pair Encoding (BPE) tokenization to the text.
        :param text: Input text
        :return: List of subword tokens
        """
        tokens = self.tokenize(text)
        bpe_tokens = []
        for token in tokens:
            sub_tokens = self._apply_bpe(token)
            bpe_tokens.extend(sub_tokens)
        return bpe_tokens

    def _apply_bpe(self, token: str) -> List[str]:
        """
        Apply BPE rules to a single token.
        :param token: Input token
        :return: List of subword tokens
        """
        # Simplified BPE example
        if token in self.vocab:
            return [token]
        sub_tokens = []
        while len(token) > 0:
            matched = False
            for i in range(len(token), 0, -1):
                sub_token = token[:i]
                if sub_token in self.vocab:
                    sub_tokens.append(sub_token)
                    token = token[i:]
                    matched = True
                    break
            if not matched:
                sub_tokens.append(token)  # handle unknown subwords
                break
        return sub_tokens

    def replace_with_synonyms(self, tokens: List[str]) -> List[str]:
        """
        Replace tokens with their synonyms based on WordNet.
        :param tokens: List of tokens
        :return: List of tokens with some replaced by synonyms
        """
        new_tokens = []
        for token in tokens:
            synonyms = wordnet.synsets(token)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()  # Choose the first synonym
                new_tokens.append(synonym)
            else:
                new_tokens.append(token)
        return new_tokens

    def create_token_id_map(self):
          """
          Create a mapping of tokens to unique IDs.
          """
          self.token_to_id = {token: idx for idx, token in enumerate(self.vocab, 1)}
          self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def get_id_from_token(self, token: str) -> int:
        """
        Get the ID for a given token.
        """
        return self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0))

    def get_token_from_id(self, token_id: int) -> str:
        """
        Get the token for a given ID.
        """
        return self.id_to_token.get(token_id, '<UNK>')
