import numpy as np
import re
import itertools
from collections import Counter
import os,gensim, logging
import string,nltk,codecs
from nltk.stem.porter import *
from nltk.corpus import wordnet as WN
from nltk.corpus import stopwords
from TwitterCleanuper import TwitterCleanuper
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_tweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to ''
    # tweet = re.sub('@[^\s]+', '', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Remove numbers
    # tweet = re.sub(r"\s?[0-9]+\.?[0-9]*", '', tweet)
    for remove in map(lambda r: re.compile(re.escape(r)), [",", "\"", "&", "%", "$",
                                                                  "%", "\\",  "?",
                                                                 "--", "---", "#"]):
        tweet = re.sub(remove, '', tweet)
    # trim
    tweet = tweet.strip('\'"')


    return tweet

# from spellchecker.spellcheck import correct

# stop_words_en = set(stopwords.words('english'))
#
#
# def tokens(sent):
#     return nltk.word_tokenize(sent)
#
#
# def spell_check(line):
#     for i in tokens(line):
#         strip = i.rstrip()
#         if not WN.synsets(strip):
#             if strip not in stop_words_en:    # <--- Check whether it's in stopword list
#                 print("Wrong spellings : " + i)
#                 cor=correct(i)
#                 print cor
#                 line=line.replace(i,cor)
#     return line

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
     #spell_check(string)


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/ADE-negative.txt").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/ADE-positive.txt").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_pos_neg(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    return [positive_examples, negative_examples]


def load_data_and_y_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]

def load_semeval_and_y_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    cleaned=[]
    y=[]
    for ex in examples:
        # if ex=='Not Available':
        #     continue
        strs=re.split(r'\t+', ex)
        cleaned.append(clean_str(strs[3].strip()))
        if strs[2]=='positive':
            y.append(1)
        elif strs[2]=='negative':
            y.append(2)
        else:
            y.append(0)


    return [cleaned, y]

def load_semeval_test(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    cleaned=[]
    for ex in examples:
        # if ex=='Not Available':
        #     continue
        strs=re.split(r'\t+', ex)
        cleaned.append(clean_str(strs[3].strip()))
    return cleaned

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]

def generateSentementFeatures(x_text,max_length):
    sequence_length = max_length#max(len(x) for x in x_text)
    stemmer = PorterStemmer()
    unstemmed_texts = []
    for line in x_text:
        # line = line.decode('utf8', 'ignore').encode('ascii', 'ignore')
        # try:
            # items = line.split('\t')
            # if len(items) > 3:
                # _drug = string.lower(string.strip(items[1]))
                # _text = string.lower(string.strip(items[-1]))
                # _class = int(string.strip(items[2]))
        senttokens = nltk.word_tokenize(line)
        stemmed_text = ''
        for t in senttokens:
            stemmed_text += ' ' + stemmer.stem(t)
        unstemmed_texts.append(line)

                # training_set['synsets'].append(FeatureExtractionUtilities.getSynsetString(_text, None))
                # print 'synsets==>'+str(training_set['synsets'])
                # training_set['clusters'].append(FeatureExtractionUtilities.getclusterfeatures(_text))
                # print 'clusters==>' + str(training_set['clusters'])
                # training_set['text'].append(stemmed_text)
                # training_set['class'].append(_class)
                # print 'class==>' + str(training_set['class'])
        # except UnicodeDecodeError:
        #     print 'please convert to correct encoding..'

    # infile.close()
    # a = np.zeros((len(x_text), 90), dtype=np.float64).tolist()
    # trained_data = np.concatenate((np.array(trained_data),a),axis=1)

    print 'Generating training set sentiment features .. '
    sentiments = FeatureExtractionUtilities.getsentimentfeatures(unstemmed_texts,sequence_length)
    return sentiments



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(positive_data_file, negative_data_file):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

