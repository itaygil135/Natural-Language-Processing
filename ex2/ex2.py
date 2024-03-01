# NLP - 67658
# Exercise 2
# Omer Mushlion 208271197
# Itay Chachy 208489732
# Itay Kahana 316385962
###########################

from re import search
import nltk
from nltk.corpus import brown
from collections import defaultdict

UNKNOWN_TAG = "NN"
START_TAG = "START_TAG"
STOP_TAG = "STOP_TAG"
DELTA_SMOOTHING_FACTOR = 1
MIN_FREQUENCY = 1
HAS_DIGIT_AND_ALPHA = "([A-Za-z]*\\d+[A-Za-z]*)+", "HAS_DIGIT_AND_ALPHA"
HAS_DIGIT_AND_DASH = "([0-9]*-[0-9]*)+", "HAS_DIGIT_AND_DASH"
HAS_DIGIT_AND_COMMA = "([0-9]*,[0-9]*)+", "HAS_DIGIT_AND_COMMA"
HAS_DIGIT_AND_PERIOD = "([0-9]*\\.[0-9]*)+", "HAS_DIGIT_AND_PERIOD"
ALL_CAPS = "[A-Z]*", "ALL_CAPS"
CAP_AND_PERIOD = "[A-Z]\\.", "CAP_PERIOD"
TWO_DIGIT_NUM = "TWO_DIGIT_NUM"
FOUR_DIGIT_NUM = "FOUR_DIGIT_NUM"
OTHER_N = "OTHER_N"
STARTS_WITH_CAP = "STARTS_WITH_CAP"
LOWER_CASED = "LOWER_CASED"
OTHER = "OTHER"


def _preprocess_tag(tag: str):
    return tag.split('+')[0].split('-')[0]


def _prepare_data(sentences, training_portion=0.9):
    processed_data = [[(word, _preprocess_tag(tag)) for word, tag in sentence] for sentence in sentences]
    index = int(len(processed_data) * training_portion) + 1
    return processed_data[:index], processed_data[index:]


def _get_baseline_model(training_data):
    tags_for_word = defaultdict(lambda: defaultdict(int))
    for sentence in training_data:
        for word, tag in sentence:
            tags_for_word[word][tag] += 1
    model = {}
    for word, tags_dict in tags_for_word.items():
        model[word] = max(tags_dict, key=tags_dict.get)
    return model


def _get_baseline_accuracy(test_data, mle_dict):
    n_known_words, n_unknown_words = 0, 0
    n_right_predictions_known_words, n_right_predictions_unknown_words = 0, 0
    for sentence in test_data:
        for word, tag in sentence:
            is_known_word = word in mle_dict
            n_right_predictions_known_words += int(is_known_word and tag == mle_dict[word])
            n_known_words += is_known_word
            n_right_predictions_unknown_words += int(not is_known_word and tag == UNKNOWN_TAG)
            n_unknown_words += not is_known_word
    known_words_accuracy = n_right_predictions_known_words / n_known_words
    unknown_words_accuracy = n_right_predictions_unknown_words / n_unknown_words
    total_known_words_accuracy = ((n_right_predictions_unknown_words + n_right_predictions_known_words) /
                                  (n_unknown_words + n_known_words))
    return known_words_accuracy, unknown_words_accuracy, total_known_words_accuracy



def _get_add_one_emission_prob_dict(training_data):
    emission_prob_dict = defaultdict(lambda: defaultdict(float))
    words_for_tag = defaultdict(int)
    for sentence in training_data:
        words_for_tag[START_TAG] += 1
        words_for_tag[STOP_TAG] += 1
        for word, tag in sentence:
            emission_prob_dict[tag][word] += 1
            words_for_tag[tag] += 1
    for tag, words in emission_prob_dict.items():
        for word, count in words.items():
            words[word] = _smooth_count(count, tag, len(words), words_for_tag)
    return emission_prob_dict


def _smooth_count(count, tag, n, words_for_tag):
    return (count + DELTA_SMOOTHING_FACTOR) / (words_for_tag[tag] + n * DELTA_SMOOTHING_FACTOR)


def _get_transition_emission_prob_dicts(training_data):
    transition_prob_dict = defaultdict(lambda: defaultdict(float))
    emission_prob_dict = defaultdict(lambda: defaultdict(float))
    for sentence in training_data:
        previous_tag = START_TAG
        for word, tag in sentence:
            emission_prob_dict[tag][word] += 1
            transition_prob_dict[previous_tag][tag] += 1
            previous_tag = tag
        transition_prob_dict[previous_tag][STOP_TAG] += 1
    return transition_prob_dict, emission_prob_dict


def _extract_all_words_set(emission_prob_dict):
    return set(word for tag in emission_prob_dict for word in emission_prob_dict[tag])


def _extract_all_tags_dict(test_data, transition_prob_dict):
    all_tags = set(tag for sentence in test_data for word, tag in sentence).union(set(transition_prob_dict.keys()))
    all_tags_dict = {tag: i for i, tag in enumerate(all_tags)}
    all_tags_dict[STOP_TAG] = len(all_tags)
    return all_tags_dict


def _add_tags(sentence):
    return [(START_TAG, START_TAG)] + sentence + [(STOP_TAG, STOP_TAG)]

def _set_prob_and_pointer(back_pointers, emission_dict, index1, index2, table, prev_tag, transition_dict, word):
    max_val, argmax = 0, None
    for tag in transition_dict:
        val = transition_dict[prev_tag][tag] * emission_dict[tag][word]
        if val > max_val:
            max_val, argmax = val, tag
    if max_val > 0:
        table[index1 + 1][index2] *= max_val
        back_pointers[index1 + 1][index2] = argmax
    else:  # Unknown
        table[index1 + 1][index2] \
            = table[index1][index2] * transition_dict[prev_tag][UNKNOWN_TAG] * emission_dict[UNKNOWN_TAG][word]
        back_pointers[index1 + 1][index2] = UNKNOWN_TAG


def _get_table_and_bp(emission_dict, transition_dict, words):
    t_rows = len(transition_dict.keys())
    t_cols = len(words) + 1
    table = [[1 for _ in range(t_rows)] for _ in range(t_cols)]
    back_pointers = [[None for _ in range(t_rows)] for _ in range(t_cols)]
    for i, word in enumerate(words):
        for j, prev_tag in enumerate(transition_dict):
            _set_prob_and_pointer(back_pointers, emission_dict, i, j, table, prev_tag, transition_dict, word)
    return table, back_pointers


def _viterbi_algorithm(transition_dict, emission_dict, words):
    table, back_pointers = _get_table_and_bp(emission_dict, transition_dict, words)
    return _extract_labels(back_pointers, table, transition_dict, words)


def _extract_labels(back_pointers, table, transition_dict, words):
    labels = [None for _ in range(len(words) + 2)]
    for i in range(len(words) - 1, -1, -1):
        max_val, max_idx = 0, 0
        for j in range(len(transition_dict)):
            prob = table[i + 1][j]
            if prob > max_val:
                max_val, max_idx = prob, j
        labels[i] = back_pointers[i + 1][max_idx]
    labels[0], labels[-1] = START_TAG, STOP_TAG
    return labels


def _compute_hmm_accuracy(test_data, transition_prob_dict, emission_prob_dict):
    n_known_words, n_unknown_words = 0, 0
    n_right_predictions_known_words, n_right_predictions_unknown_words = 0, 0
    all_words = _extract_all_words_set(emission_prob_dict)
    all_tags_dict = _extract_all_tags_dict(test_data, transition_prob_dict)

    n = len(all_tags_dict.keys())
    confusion_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for sentence in test_data:
        sentence = _add_tags(sentence)
        words, labels = [t[0] for t in sentence], [t[1] for t in sentence]
        predicted_labels = _viterbi_algorithm(transition_prob_dict, emission_prob_dict, words)
        for word, label, predicted_label in zip(words, labels, predicted_labels):
            confusion_matrix[all_tags_dict[predicted_label]][all_tags_dict[label]] += 1
            if word in all_words:
                n_right_predictions_known_words += (predicted_label == label)
                n_known_words += 1
            else:
                n_right_predictions_unknown_words += (predicted_label == label)
                n_unknown_words += 1
    known_accuracy = (n_right_predictions_known_words / n_known_words)
    unknown_accuracy = (n_right_predictions_unknown_words / n_unknown_words)
    total_accuracy = ((n_right_predictions_unknown_words + n_right_predictions_known_words) /
                            (n_unknown_words + n_known_words))
    return (known_accuracy, unknown_accuracy, total_accuracy), confusion_matrix


def _modify_data_with_pseudo_words(training_data, test_data):
    word_counter = defaultdict(int)
    for sentence in training_data:
        for word, tag in sentence:
            word_counter[word] += 1
    new_training_data = _update_data(training_data, word_counter, is_test_data=False)
    new_test_data = _update_data(test_data, word_counter, is_test_data=True)
    return new_training_data, new_test_data


def _update_data(data, word_counter, is_test_data):
    new_data = []
    for sentence in data:
        new_sentence = []
        for word, tag in sentence:
            if is_test_data and word not in word_counter:
                new_sentence.append((_get_pseudo_word(word), tag))
            elif not is_test_data and word_counter[word] <= MIN_FREQUENCY:
                new_sentence.append((_get_pseudo_word(word), tag))
            else:
                new_sentence.append((word, tag))
        new_data.append(new_sentence)
    return new_data


def _get_pseudo_word(word: str):
    if word.isnumeric() and len(word) == 2:
        pseudo_word = TWO_DIGIT_NUM
    elif word.isnumeric() and len(word) == 4:
        pseudo_word = FOUR_DIGIT_NUM
    elif word.isnumeric():
        pseudo_word = OTHER_N
    elif search(HAS_DIGIT_AND_ALPHA[0], word):
        pseudo_word = HAS_DIGIT_AND_ALPHA[1]
    elif search(HAS_DIGIT_AND_DASH[0], word) and word != '-':
        pseudo_word = HAS_DIGIT_AND_DASH[0]
    elif search(HAS_DIGIT_AND_COMMA[0], word) and word != ',':
        pseudo_word = HAS_DIGIT_AND_COMMA[1]
    elif search(HAS_DIGIT_AND_PERIOD[0], word) and word != '.':
        pseudo_word = HAS_DIGIT_AND_PERIOD[1]
    elif search(ALL_CAPS[0], word):
        pseudo_word = ALL_CAPS[1]
    elif search(CAP_AND_PERIOD[0], word):
        pseudo_word = CAP_AND_PERIOD[0]
    elif word == word.capitalize():
        pseudo_word = STARTS_WITH_CAP
    elif word.islower():
        pseudo_word = LOWER_CASED
    else:
        pseudo_word = OTHER
    return pseudo_word


if __name__ == '__main__':
    nltk.download("brown")

    # a
    tagged_sentences = brown.tagged_sents(categories='news')
    training_data, test_data = _prepare_data(tagged_sentences)

    # b
    baseline_accuracy = _get_baseline_accuracy(test_data, _get_baseline_model(training_data))
    print()
    print("b ii")
    print(f"Error rate for the baseline model for known words is: {1 - baseline_accuracy[0]}")
    print(f"Error rate for the baseline model for unknown words is: {1 - baseline_accuracy[1]}")
    print(f"Error rate for the baseline model at total: {1 - baseline_accuracy[2]}")

    # c
    transition, emission = _get_transition_emission_prob_dicts(training_data)
    hmm_accuracy = _compute_hmm_accuracy(test_data, transition, emission)[0]

    print()
    print("c iii")
    print(f"Error rate for the hmm model for known words is: {1 - hmm_accuracy[0]}")
    print(f"Error rate for the hmm model for unknown words is: {1 - hmm_accuracy[1]}")
    print(f"Error rate for the hmm model at total: {1 - hmm_accuracy[2]}")

    # d
    add_one_emission = _get_add_one_emission_prob_dict(training_data)
    hmm_add_one_accuracy = _compute_hmm_accuracy(test_data, transition, add_one_emission)[0]

    print()
    print("d ii")
    print(f"Error rate for the hmm ade-one model for known words is: {1 - hmm_add_one_accuracy[0]}")
    print(f"Error rate for the hmm ade-one model for unknown words is: {1 - hmm_add_one_accuracy[1]}")
    print(f"Error rate for the hmm ade-one model at total: {1 - hmm_add_one_accuracy[2]}")

    # e
    print()
    print("e iii")
    training_data_with_pseudo, test_data_with_pseudo = _modify_data_with_pseudo_words(training_data, test_data)
    transition_with_pseudo, emission_with_pseudo = _get_transition_emission_prob_dicts(training_data_with_pseudo)
    hmm_with_pseudo_accuracy = _compute_hmm_accuracy(test_data_with_pseudo, transition_with_pseudo, emission_with_pseudo)[0]
    print()
    print("hmm with pseudo")
    print(f"Error rate for the hmm with pseudo words model for known words is: {1 - hmm_with_pseudo_accuracy[0]}")
    print(f"Error rate for the hmm with pseudo words  model for unknown words is: {1 - hmm_with_pseudo_accuracy[1]}")
    print(f"Error rate for the hmm with pseudo words  model at total: {1 - hmm_with_pseudo_accuracy[2]}")

    add_one_emission_with_pseudo = _get_add_one_emission_prob_dict(training_data_with_pseudo)
    hmm_add_one_with_pseudo_accuracy, confusion_matrix = _compute_hmm_accuracy(test_data_with_pseudo, transition_with_pseudo, add_one_emission_with_pseudo)
    print()
    print("hmm with pseudo and add one")
    print(f"Error rate for the hmm with pseudo and add one for known words is: {1 - hmm_add_one_with_pseudo_accuracy[0]}")
    print(f"Error rate for the hmm with pseudo and add one for unknown words is: {1 - hmm_add_one_with_pseudo_accuracy[1]}")
    print(f"Error rate for the hmm with pseudo and add one at total: {1 - hmm_add_one_with_pseudo_accuracy[2]}")
    print("confusion_matrix for hmm with pseudo and add one:")
    print(confusion_matrix)
