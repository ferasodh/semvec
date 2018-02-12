#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Mar 19, 2016

@author: asarker

This is a binary classifier for social media posts. The classifier learns to distinguish between posts that mention 
adverse drug reactions and those that don't.

The performance of this SVM classifier varies with the kernel, the cost parameter and the weights. 

Last best run used the parameters:
kernel: RBF
cost (c): 140
weight: 3

Please run 10-fold cross validation with a range of values to optimize classifier for a new data set.

***THIRD PARTY RESOURCES***
The classifier utilizes a number of third party resources. 


SENTIMENT SCORES
1. POSITIVE AND NEGATIVE TERMS
AVAILABLE PUBLICLY AT: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

CITATION:
Minqing Hu and Bing. 2004. ”mining and summarizing customer reviews”. In Proceedings of the
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

2. PRIOR POLARITIES
AVAILABLE PUBLICLY AT: https://hlt-nlp.fbk.eu/technologies/sentiwords
CITATION:
Marco Guerini, Lorenzo Gatti, and Marco Turchi. 2013. Sentiment Analysis: How to Derive Prior Polarities
from SentiWordNet. In Proceedings of Empirical Methods in Natural Language Processing (EMNLP), pages 1259–1269.

3. MULTI-PERSPECTIVE QUESTION ANSWERING
AVAILABLE AT: http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
LICENSE: GNU public license: http://www.gnu.org/licenses/gpl.html


TWITTER WORD CLUSTERS
filename: 50mpaths2.txt
Available at: http://www.cs.cmu.edu/~ark/TweetNLP/
CITATION: Olutobi Owoputi, Brendan O’Connor, Chris Dyer Kevin Gimpel, and Nathan Schneider. 2012. Part-of-Speech Tagging for Twitter: Word Clusters and Other Ad-
vances. Technical report, School of Computer Science, Carnegie Mellon University.
***************************


'''

import string,nltk,codecs

from keras.layers import Convolution2D
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from nltk.corpus import stopwords
from sklearn import svm
from collections import defaultdict
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
import keras
import data_helpers
from w2v import train_word2vec
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D,MaxPooling2D
# from keras.utils.visualize_util import plot
# from keras.utils.visualize_util import model_to_dot
from keras.callbacks import TensorBoard
np.random.seed(2)


stemmer = PorterStemmer()


def loadFeatureExtractionModuleItems():
    FeatureExtractionUtilities.loadItems()

if __name__ == '__main__':
    global good_words
    global bad_words
    global more_words
    global less_words
    global senti_words
    global modals
    global topic_dict
    loadFeatureExtractionModuleItems()
    
    training_set = defaultdict(list)
    tb=TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
    
    '''
        Generate the training set features...
    '''
    #work with a sample..
    infile = open('./twitter_binary')

    data_size = 10


    
    #infile = open('./twitter_binary_training_set.txt')
    unstemmed_texts = []
    for line in infile:
        line = line.decode('utf8','ignore').encode('ascii','ignore')
        try:
            items = line.split('\t')
            if len(items)>3:
                _drug = string.lower(string.strip(items[1]))
                _text = string.lower(string.strip(items[-1]))
                _class = int(string.strip(items[2]))
                senttokens = nltk.word_tokenize(_text)
                stemmed_text = ''
                for t in senttokens:
                    stemmed_text += ' ' + stemmer.stem(t)
                unstemmed_texts.append(_text)
               
                # training_set['synsets'].append(FeatureExtractionUtilities.getSynsetString(_text, None))
                # print 'synsets==>'+str(training_set['synsets'])
                # training_set['clusters'].append(FeatureExtractionUtilities.getclusterfeatures(_text))
                # print 'clusters==>' + str(training_set['clusters'])
                training_set['text'].append(stemmed_text)
                training_set['class'].append(_class)
                # print 'class==>' + str(training_set['class'])
        except UnicodeDecodeError:
            print 'please convert to correct encoding..'
    
    infile.close()
    

    print 'Generating training set sentiment features .. '
    training_set['sentiments'] = FeatureExtractionUtilities.getsentimentfeatures(unstemmed_texts)
    # print 'training_set[sentiments]'+str(training_set['sentiments'])
    # training_set['structuralfeatures'] = FeatureExtractionUtilities.getstructuralfeatures(unstemmed_texts)
    # print 'training_set[structuralfeatures]'+str(training_set['structuralfeatures'])
    # scaler1 = preprocessing.StandardScaler().fit( training_set['structuralfeatures'])
    # train_structural_features = scaler1.transform( training_set['structuralfeatures'])
    # training_set['adrlexicon'] = FeatureExtractionUtilities.getlexiconfeatures(unstemmed_texts)
    # print 'adrlexicon==>' + str(training_set['adrlexicon'])
    # training_set['topictexts'],training_set['topics'] = FeatureExtractionUtilities.gettopicscores(training_set['text'])
    # print 'topictexts==>' + str(training_set['topictexts'])
    # training_set['goodbad'] = FeatureExtractionUtilities.goodbadFeatures(training_set['text'])
    # print 'goodbad==>' + str(training_set['goodbad'])
    '''
        Initialize the vectorizers
    '''
    print 'Initialize the vectorizers..'
    synsetvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 2000)
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer = "word", tokenizer = None, preprocessor = None, max_features = 5000)
    clustervectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 1000)
    topicvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features=500)
    
    '''
        Generate the data vectors
    '''
    print 'Generate the data vectors'
    # trained_data = vectorizer.fit_transform(training_set['text']).toarray()

    # train_data_synset_vector = synsetvectorizer.fit_transform(training_set['synsets']).toarray()
    # train_data_cluster_vector = clustervectorizer.fit_transform(training_set['clusters']).toarray()
    # print 'trained_data'
    # print trained_data[0]
    # train_data_topic_vector = topicvectorizer.fit_transform(training_set['topictexts']).toarray()
    
    '''
        Concatenate the various feature arrays
    '''
    print 'Concatenate the various feature arrays'
    # trained_data = np.concatenate((trained_data,train_data_synset_vector),axis=1)
    trained_data =training_set['sentiments']  #np.concatenate((trained_data,training_set['sentiments']),axis=1)
    # trained_data = np.concatenate((trained_data,train_data_cluster_vector),axis=1)
    # trained_data = np.concatenate((trained_data,train_structural_features),axis=1)
    # trained_data = np.concatenate((trained_data,training_set['adrlexicon']),axis=1)
    # trained_data = np.concatenate((trained_data,training_set['topics']),axis=1)
    # trained_data = np.concatenate((trained_data,train_data_topic_vector),axis=1)
    # trained_data = np.concatenate((trained_data,training_set['goodbad']),axis=1)
    # a=np.zeros((data_size,90), dtype=np.float64).tolist()
    # trained_data = np.concatenate((np.array(trained_data),a),axis=1)

    maxLen= max(len(x) for x in trained_data)
    # rowsLen=maxLen
    # while maxLen> rowsLen *30:
    #     rowsLen+=1
    # maxLen=rowsLen*30

    # print trained_data[3].shape
    # print maxLen
    x=training_set['sentiments']
    # for row in trained_data:
    #     rem=maxLen-len(row)
    #     if rem>0:
    #         a = np.zeros((rem,), dtype=np.float64)
    #         row=np.concatenate((row,a),axis=0)
    #     # print len(row)
    #     # tmp=np.array(row, np.float32).reshape(1,rowsLen, 30)
    #     x.append(row)
    # print x[0].shape

    # Model Hyperparameters
    sequence_length = 154
    embedding_dim = 150
    filter_sizes = (2,3,4)
    num_filters = 2
    dropout_prob = (0.25, 0.5)
    hidden_dims = 150
    column_size=30


    # Training parameters
    batch_size = 32
    num_epochs = 100
    val_split = 0.1

    # Word2Vec parameters, see train_word2vec
    min_word_count = 1  # Minimum word count
    context = 7  # Context window size

    # Data Preparatopn
    # ==================================================
    #
    # Load data
    print("Loading data...")
    # x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
    #
    # if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    #     embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    #     if model_variation == 'CNN-static':
    #         x = embedding_weights[0][x]
    # elif model_variation == 'CNN-rand':
    #     embedding_weights = None
    # else:
    #     raise ValueError('Unknown model variation')

    pool_size = (3, 5)
        # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(data_size - 1))
    x_shuffled = np.array(x)[shuffle_indices]#np.array(x)#[shuffle_indices]
    y_shuffled = np.array(training_set['class'])[shuffle_indices]#[shuffle_indices].argmax(axis=1)
    # print x_shuffled.shape


    # print("Vocabulary Size: {:d}".format(len(vocabulary)))

    # Building model
    # ==================================================
    #
    # graph subnet with one input and one output,
    # convolutional layers concateneted in parallel
    graph_in = Input(shape=(9,20,20))
    # #
    convs = []
    for fsz in filter_sizes:
        # conv = Convolution1D(nb_filter=num_filters,
        #                      filter_length=fsz,
        #                      border_mode='valid',
        #                      activation='relu')(graph_in)#subsample_length=1,input_dim=1,
        # pool = MaxPooling1D(pool_length=2)(conv)
        conv=Convolution2D(2,5,20)(graph_in)
        pool=MaxPooling2D(pool_size=pool_size)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    # # #
    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]
    #
    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    # model.add(Dense(1, batch_input_shape=(None, 1010),init='uniform', activation='sigmoid'))
    # if not model_variation == 'CNN-static':
    #     model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
    #                         weights=embedding_weights))

    # # model.add(Dropout(dropout_prob[0], input_shape=(sequence_length,data_size)))
    model.add(graph)
    # model.add(Dense(hidden_dims))
    # # model.add(LSTM(output_dim=150, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(dropout_prob[1]))
    #
    # model.add(Activation('relu'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop',
    #
    # model.add(Convolution2D(64,15,15, border_mode='full', input_shape=(1,rowsLen, 30)))
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(64, 3,3))
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    #
    # model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, rowsLen, 30)))(graph_in)
    # model.add(Activation('relu'))
    # # # model.add(Convolution2D(32, 3, 3))
    # # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    # #
    # model.add(Flatten())
    # model.add(graph)
    # Note: Keras does automatic shape inference.
    # model.add(Dense(16))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # model.add(Dense(5, input_dim=5, init='uniform', activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(2, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', 'fmeasure', 'precision', 'recall'])
    model.summary()

    # Training model
    # ==================================================
    model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
              nb_epoch=num_epochs, validation_split=val_split, verbose=2,callbacks=[tb])
