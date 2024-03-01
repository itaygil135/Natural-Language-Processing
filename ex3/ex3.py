# NLP - 67658
# Exercise 3
# Omer Mushlion 208271197
# Itay Chachy 208489732
# Itay Kahana 316385962
###########################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle
from tqdm import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
BATCH_SIZE = 64
LOG_LINEAR_EPOCHS = 20
LSTM_EPOCHS = 4
LSTM_HIDDEN_SIZE = 100
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_vectors = [word_to_vec[word] if word in word_to_vec else np.zeros(embedding_dim) for word in sent.text]
    if not w2v_vectors:
        return np.zeros(embedding_dim)
    return np.mean(w2v_vectors, axis=0)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot_vector = np.zeros(size)
    one_hot_vector[ind] = 1.
    return one_hot_vector


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vocabulary_size = len(word_to_ind)
    one_hot_vectors = [get_one_hot(vocabulary_size, word_to_ind[w]) for w in sent.text]
    return np.mean(one_hot_vectors, axis=0)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {w: i for i, w in enumerate(set(words_list))}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embedding = np.zeros((seq_len, embedding_dim))
    for i, word in zip(range(seq_len), sent.text):
        if word in word_to_vec:
            embedding[i] = word_to_vec[word]
    return embedding


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            dtype=torch.float32
        )
        self.linear = nn.Linear(2 * hidden_dim, 1, dtype=torch.float32)

    def forward(self, text):
        lstm_output, _ = self.lstm(text)
        hidden_1 = lstm_output[:, -1, :self.hidden_dim]
        hidden_2 = lstm_output[:, 0, self.hidden_dim:]
        hidden_concatenated = torch.cat((hidden_1, hidden_2), dim=1)
        return self.linear(hidden_concatenated)

    def predict(self, text):
        return torch.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1, dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.sigmoid(self.linear(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    preds = torch.round(torch.sigmoid(preds))
    return torch.mean(preds == y, dtype=torch.float32)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    loss = 0.
    accuracy = 0.
    for batch_idx, (data, labels) in enumerate(data_iterator):
        optimizer.zero_grad()
        output = model(data.float()).squeeze()
        batch_loss = criterion(output, labels)
        batch_accuracy = binary_accuracy(output, labels)
        accuracy += batch_accuracy
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
    return accuracy / len(data_iterator), loss / len(data_iterator)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models.
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average accuracy over all examples, average loss over all examples)
    """
    model.eval()
    loss = 0.
    accuracy = 0.
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_iterator):
            output = model(data.float()).squeeze()
            batch_loss = criterion(output, labels)
            batch_accuracy = binary_accuracy(output, labels)
            accuracy += batch_accuracy
            loss += batch_loss.item()
    return accuracy / len(data_iterator), loss / len(data_iterator)


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in data_iter:
            output = model.predict(data.float())
            predictions.append(output)
    return torch.cat(predictions, dim=0)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_data_iter = data_manager.get_torch_iterator(data_subset=TRAIN)
    validation_data_iter = data_manager.get_torch_iterator(data_subset=VAL)
    criterion = nn.BCEWithLogitsLoss()
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    for _ in tqdm(range(n_epochs)):
        epoch_training_accuracy, epoch_training_loss = train_epoch(model, training_data_iter, optimizer, criterion)
        training_accuracy.append(epoch_training_accuracy)
        training_loss.append(epoch_training_loss)
        epoch_validation_accuracy, epoch_validation_loss = evaluate(model, validation_data_iter, criterion)
        validation_accuracy.append(epoch_validation_accuracy)
        validation_loss.append(epoch_validation_loss)
    return model, training_accuracy, training_loss, validation_accuracy, validation_loss, criterion


def get_negated_and_rare_data_loaders(data_manager: DataManager, batch_size=64):
    sentences = data_manager.sentences[TEST]
    basic_dataset = data_manager.torch_datasets[TEST]

    neg = data_loader.get_negated_polarity_examples(sentences)
    neg_set = torch.utils.data.Subset(basic_dataset, neg)
    neg_loader = torch.utils.data.DataLoader(neg_set, batch_size=batch_size, shuffle=False)

    rare = data_loader.get_rare_words_examples(sentences, data_manager.sentiment_dataset)
    rare_set = torch.utils.data.Subset(basic_dataset, rare)
    rare_loader = torch.utils.data.DataLoader(rare_set, batch_size=batch_size, shuffle=False)
    return neg_loader, rare_loader


def plot_training_accuracy(title, training_accuracy, validation_accuracy):
    epoch = np.arange(1, len(training_accuracy) + 1)
    plt.plot(epoch, training_accuracy, label="Training Accuracy")
    plt.plot(epoch, validation_accuracy, label="Validation Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_training_loss(title, training_loss, validation_loss):
    epoch = np.arange(1, len(training_loss) + 1)
    plt.plot(epoch, training_loss, label="Training Loss")
    plt.plot(epoch, validation_loss, label="Validation Loss")
    plt.title(title)
    plt.legend()
    plt.show()

def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    model = LogLinear(data_manager.get_input_shape()[0])
    model, training_accuracy, training_loss, validation_accuracy, validation_loss, criterion = (
        train_model(model, data_manager, n_epochs=LOG_LINEAR_EPOCHS, lr=0.01, weight_decay=0.001)
    )

    title = "Log-Linear - Average \"one hot encoding\" - Accuracy"
    plot_training_accuracy(title, training_accuracy, validation_accuracy)

    title = "Log-Linear - Average \"one hot encoding\" - Loss"
    plot_training_loss(title, training_loss, validation_loss)

    test_data_iter = data_manager.get_torch_iterator(data_subset=TEST)
    test_accuracy, test_loss = evaluate(model, test_data_iter, criterion)
    print(f"Log-Linear - Average \"one hot encoding\" - Test accuracy: {test_accuracy:.3f}")
    print(f"Log-Linear - Average \"one hot encoding\" - Test Loss: {test_loss:.3f}")

    neg_loader, rare_loader = get_negated_and_rare_data_loaders(data_manager, BATCH_SIZE)
    neg_accuracy, _ = evaluate(model, neg_loader, criterion)
    rare_accuracy, _ = evaluate(model, rare_loader, criterion)

    print(f"Log-Linear - Average \"one hot encoding\" - Negated accuracy: {neg_accuracy:.3f}")
    print(f"Log-Linear - Average \"one hot encoding\" - Rare accuracy: {rare_accuracy:.3f}")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(data_manager.get_input_shape()[0])
    model, training_accuracy, training_loss, validation_accuracy, validation_loss, criterion = (
        train_model(model, data_manager, n_epochs=LOG_LINEAR_EPOCHS, lr=0.01, weight_decay=0.001)
    )

    title = "Log-Linear - Average \"Word2Vec\" - Accuracy"
    plot_training_accuracy(title, training_accuracy, validation_accuracy)

    title = "Log-Linear - Average \"Word2Vec\" - Loss"
    plot_training_loss(title, training_loss, validation_loss)

    test_data_iter = data_manager.get_torch_iterator(data_subset=TEST)
    test_accuracy, test_loss = evaluate(model, test_data_iter, criterion)
    print(f"Log-Linear - Average \"Word2Vec\" - Test accuracy: {test_accuracy:.3f}")
    print(f"Log-Linear - Average \"Word2Vec\" - Test Loss: {test_loss:.3f}")

    neg_loader, rare_loader = get_negated_and_rare_data_loaders(data_manager, BATCH_SIZE)
    neg_accuracy, _ = evaluate(model, neg_loader, criterion)
    rare_accuracy, _ = evaluate(model, rare_loader, criterion)

    print(f"Log-Linear - Average \"Word2Vec\" - Negated accuracy: {neg_accuracy:.3f}")
    print(f"Log-Linear - Average \"Word2Vec\" - Rare accuracy: {rare_accuracy:.3f}")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=LSTM_HIDDEN_SIZE, dropout=0.5)
    model, training_accuracy, training_loss, validation_accuracy, validation_loss, criterion = (
        train_model(model, data_manager, n_epochs=LSTM_EPOCHS, lr=0.001, weight_decay=0.0001)
    )

    title = "LSTM - \"Word2Vec\" - Accuracy"
    plot_training_accuracy(title, training_accuracy, validation_accuracy)

    title = "LSTM - \"Word2Vec\" - Loss"
    plot_training_loss(title, training_loss, validation_loss)

    test_data_iter = data_manager.get_torch_iterator(data_subset=TEST)
    test_accuracy, test_loss = evaluate(model, test_data_iter, criterion)
    print(f"LSTM - \"Word2Vec\" - Test accuracy: {test_accuracy:.3f}")
    print(f"LSTM - \"Word2Vec\" \"Word2Vec\" - Test Loss: {test_loss:.3f}")

    neg_loader, rare_loader = get_negated_and_rare_data_loaders(data_manager, BATCH_SIZE)
    neg_accuracy, _ = evaluate(model, neg_loader, criterion)
    rare_accuracy, _ = evaluate(model, rare_loader, criterion)

    print(f"LSTM - \"Word2Vec\" - Negated accuracy: {neg_accuracy:.3f}")
    print(f"LSTM - \"Word2Vec\" - Rare accuracy: {rare_accuracy:.3f}")


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
