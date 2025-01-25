import re
from collections import Counter, defaultdict
import numpy as np

class NGramModel:
    def __init__(self, n):
        """
        Initialize the N-gram model with specified n value.
        :param n: The size of N for the N-grams (e.g., 2 for bigrams, 3 for trigrams).
        """
        self.n = n
        self.ngrams = Counter()
        self.context_counts = Counter()

    def preprocess_text(self, text):
        """
        Preprocess the input text: lowercase, remove punctuation, tokenize.
        :param text: Raw input string.
        :return: List of tokens.
        """
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def generate_ngrams(self, tokens):
        """
        Generate N-grams from a list of tokens.
        :param tokens: List of tokens.
        :return: List of N-grams.
        """
        if len(tokens) < self.n:
            return []
        return [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]

    def fit(self, corpus):
        """
        Fit the model on a corpus of text.
        :param corpus: A list of sentences or a large text corpus.
        """
        for sentence in corpus:
            tokens = self.preprocess_text(sentence)
            ngrams = self.generate_ngrams(tokens)
            self.ngrams.update(ngrams)
            for ngram in ngrams:
                self.context_counts[ngram[:-1]] += 1

    def predict_next_word(self, context):
        """
        Predict the next word given a context.
        :param context: Tuple of words as context (length = n-1).
        :return: Most probable next word.
        """
        if len(context) != self.n - 1:
            raise ValueError(f"Context length must be {self.n - 1}.")

        candidates = {ngram[-1]: count for ngram, count in self.ngrams.items() if ngram[:-1] == context}
        total_count = sum(candidates.values())
        if total_count == 0:
            return None  # No candidates found
        probabilities = {word: count / total_count for word, count in candidates.items()}
        return max(probabilities, key=probabilities.get)

    def generate_sentence(self, start_words, max_length=20):
        """
        Generate a sentence starting with given words.
        :param start_words: List of words to start the sentence.
        :param max_length: Maximum length of the generated sentence.
        :return: Generated sentence as a string.
        """
        sentence = list(start_words)
        for _ in range(max_length - len(start_words)):
            context = tuple(sentence[-(self.n - 1):])
            next_word = self.predict_next_word(context)
            if next_word is None:
                break
            sentence.append(next_word)
        return ' '.join(sentence)

    def perplexity(self, corpus):
        """
        Calculate the perplexity of the model on a given corpus.
        :param corpus: A list of sentences or a text corpus.
        :return: Perplexity score.
        """
        total_log_prob = 0
        total_words = 0
        for sentence in corpus:
            tokens = self.preprocess_text(sentence)
            ngrams = self.generate_ngrams(tokens)
            for ngram in ngrams:
                context, target = ngram[:-1], ngram[-1]
                prob = self._ngram_probability(context, target)
                total_log_prob += np.log(prob) if prob > 0 else float('-inf')
                total_words += 1
        return np.exp(-total_log_prob / total_words)

    def _ngram_probability(self, context, word):
        """
        Helper function to calculate the probability of a word given its context.
        :param context: Tuple of words (context).
        :param word: Target word.
        :return: Probability.
        """
        context_count = self.context_counts[context]
        if context_count == 0:
            return 0  # Avoid division by zero
        ngram_count = self.ngrams[context + (word,)]
        return ngram_count / context_count if context_count > 0 else 0
# Example corpus
corpus = [
    "The quick brown fox jumps over the lazy cat, but she likes milk",
    "The quick blue fox jumps high",
    "The fox is quick and smart"
]

# Instantiate and train a trigram model
ngram_model = NGramModel(n=5)
ngram_model.fit(corpus)

# Generate a sentence
print("Generated sentence:", ngram_model.generate_sentence(["the", "fox", "is", "quick"], max_length=13))

# Calculate perplexity on the same corpus
print("Perplexity:", ngram_model.perplexity(corpus))
