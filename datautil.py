import unidecode
import torch as tr
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from torchtext.data.utils import get_tokenizer
import random
import csv

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"@\S+|http\S+|www\S+|https\S+|\S+@\S+|\S*\d+\S*", "", text)

    # Handle emojis
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

def tokenize_text(text, max_token_len=12, min_token_len=1):
    stop_words = set(stopwords.words('english'))
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    tokens = [token for token in tokens if token not in stop_words and not None and len(token) <= max_token_len and len(token) >= min_token_len]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def build_vocab(tokens):
    return set(tokens)

def padded_batch(batch):
    max_length = max(len(seq) for seq in batch)
    padded_batch = [tr.cat([seq, tr.zeros(max_length - len(seq), dtype=tr.int)]) for seq in batch]
    return tr.stack(padded_batch)

def build_dataset():
    # load data
    with open('data/labeled_data.csv') as data:
        csv_data = [row for row in csv.DictReader(data)]
    
    # Define the ratio for train, test, and validation sets
    train_ratio = 0.7
    test_ratio = 0.15
    num_samples = len(csv_data)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = int(test_ratio * num_samples)

    indices = list(range(num_samples))
    random.shuffle(indices)

    train_data = [csv_data[i] for i in indices[:num_train_samples]]
    test_data = [csv_data[i] for i in indices[num_train_samples:num_train_samples + num_test_samples]]
    val_data = [csv_data[i] for i in indices[num_train_samples + num_test_samples:]]

    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))
    print("Validation data size:", len(val_data))

    return train_data, test_data, val_data

def create_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def line_to_indices(word_to_ix, line):
    tokens = tokenize_text(preprocess_text(line))
    return tr.tensor([word_to_ix.get(w, word_to_ix['<unk>']) for w in tokens]).int()

def input_output(word_to_ix, batch):
    xb = [line_to_indices(word_to_ix, row['tweet']) for row in batch]
    xb = padded_batch(xb)
    xb = xb.view(xb.shape[0], 1, xb.shape[1])
    yb = tr.tensor([int(row['class']) for row in batch])
    return xb, yb

def get_word_to_ix(data):
    text = ''.join([row['tweet'] for row in data])
    text = preprocess_text(text)
    tokens = ['<unk>'] + tokenize_text(text)
    vocab = build_vocab(tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix

def get_batches_split(batch_size):
    train_data, test_data, val_data = build_dataset()
    word_to_ix = get_word_to_ix(train_data)
    train_batches = [input_output(word_to_ix, b) for b in create_batches(train_data, batch_size)]
    test_batches = [input_output(word_to_ix, b) for b in create_batches(test_data, batch_size)]
    val_batches = [input_output(word_to_ix, b) for b in create_batches(val_data, batch_size)]

    print("Train batch count:", len(train_batches))
    print("Test batch count:", len(test_batches))
    print("Validation batch count:", len(val_batches))

    return train_batches, val_batches, test_batches, word_to_ix