# NLP - 67658
# Exercise 1
# Omer Mushlion 208271197
# Itay Chachy 208489732
# Itay Kahana 316385962
###########################

from spacy import load
from datasets import load_dataset
from collections import defaultdict
import math

START = "0"


class Unigram:
    _word_counter = defaultdict(int)
    _total_number_of_words = 0

    def update(self, word):
        self._word_counter[word] += 1
        self._total_number_of_words += 1

    def get_prob_for_word(self, word):
        if self._total_number_of_words == 0:
            return -math.inf
        prob = self._word_counter[word] / self._total_number_of_words
        return prob


class Bigram:
    _single_previous_word_counter = defaultdict(int)
    _couple_words_counter = defaultdict(int)

    def update(self, cur, prev):
        self._single_previous_word_counter[prev] += 1
        self._couple_words_counter[(prev, cur)] += 1

    def get_prob_for_word(self, word, prev):
        if prev not in self._single_previous_word_counter:
            return 0
        return self._couple_words_counter[(prev, word)] / self._single_previous_word_counter[prev]

    def get_prob_for_sentence(self, sentence):
        prob = 1
        for i in range(len(sentence) - 1):
            prob *= self.get_prob_for_word(sentence[i + 1], sentence[i])
        return math.log(prob) if prob > 0 else -math.inf

    def predict_next_word(self, prev):
        next_word, m = "", 0
        for pair, n in self._couple_words_counter.items():
            if pair[0] == prev and n > m:
                next_word = pair[1]
                m = n
        return next_word


def _update_unigram(unigram, sentence):
    for word in sentence:
        unigram.update(word)


def _update_bigram(bigram, sentence):
    sentence = [START] + sentence
    for i in range(len(sentence) - 1):
        prev = sentence[i]
        cur = sentence[i + 1]
        bigram.update(cur, prev)


def _preprocess_sentence(nlp, sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if token.is_alpha]


def _set_models(unigram, bigram, vocabulary, nlp, dataset):
    for sentence in dataset["text"]:
        processed_sentence = _preprocess_sentence(nlp, sentence)
        vocabulary.update(processed_sentence)
        if len(processed_sentence) > 0:
            _update_unigram(unigram, processed_sentence)
            _update_bigram(bigram, processed_sentence)


def _evaluate_bigram(bigram, nlp, sentence):
    processed_sentence = _preprocess_sentence(nlp, sentence)
    return bigram.get_prob_for_sentence([START] + processed_sentence)


def _evaluate_interpolation(bigram, unigram, nlp, sentence, lambda_b, lambda_u):
    processed_sentence = [START] + _preprocess_sentence(nlp, sentence)
    prob = 1
    for i in range(1, len(processed_sentence)):
        p_b = bigram.get_prob_for_word(processed_sentence[i], processed_sentence[i - 1])
        p_u = unigram.get_prob_for_word(processed_sentence[i])
        prob *= (lambda_b * p_b + lambda_u * p_u)
    return math.log(prob) if prob > 0 else -math.inf


def _evaluate_perplexity(probabilities, m):
    l = sum(probabilities) / m
    return math.e ** (-l)


if __name__ == '__main__':
    nlp = load("en_core_web_sm")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    vocabulary = set()
    unigram = Unigram()
    bigram = Bigram()
    _set_models(unigram, bigram, vocabulary, nlp, dataset)

    print("\nQuestion 2")
    sentence = "I have a house in"
    preprocessed_sentence = _preprocess_sentence(nlp, sentence)
    next_word = bigram.predict_next_word(preprocessed_sentence[-1])
    print(f"The word with the highest probability to continue the sentence:'{sentence}' is: `{next_word}`")

    print("\nQuestion 3")
    sentence1 = "Brad Pitt was born in Oklahoma"
    bigram_prob1 = _evaluate_bigram(bigram, nlp, sentence1)
    print(f"The log-probability of the sentence: `{sentence1}` is {bigram_prob1}")

    sentence2 = "The actor was born in USA"
    bigram_prob2 = _evaluate_bigram(bigram, nlp, sentence2)
    print(f"The log-probability of the sentence: `{sentence2}` is {bigram_prob2}")

    m = len(sentence1.split()) + len(sentence2.split())
    perplexity = _evaluate_perplexity([bigram_prob1, bigram_prob2], m)
    print(f"The perplexity of the sentences is: {perplexity}")

    print("\nQuestion 4")
    lambda_bigram = 2 / 3
    lambda_unigram = 1 / 3

    prob1_interpolated = _evaluate_interpolation(bigram, unigram, nlp, sentence1, lambda_bigram, lambda_unigram)
    print(f"The log-probability using linear interpolation for: `{sentence1}` is {prob1_interpolated}")

    prob2_interpolated = _evaluate_interpolation(bigram, unigram, nlp, sentence2, lambda_bigram, lambda_unigram)
    print(f"The log-probability using linear interpolation for: `{sentence2}` is {prob2_interpolated}")

    perplexity_interpolated = _evaluate_perplexity([prob1_interpolated, prob2_interpolated], m)
    print(f"The perplexity of the sentences using linear interpolation is: {perplexity_interpolated}")
