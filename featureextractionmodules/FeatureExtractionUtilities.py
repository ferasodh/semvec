#!/usr/bin/env python
# -*- coding: utf-8 -*-
from featureextractionmodules import twokenize
# from cryptography.x509 import SubjectAlternativeName
from scipy.spatial.distance import cosine
from numpy import nan_to_num
from gensim.models import word2vec
# from __builtin__ import None
__author__ = 'abeedsarker'
import itertools
from functools import partial
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
import nltk, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from sklearn import svm
import cupy as np
# import gloS
import json
import re
# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, porter
from nltk.stem.porter import *
# import aspell
import scipy.spatial.distance
import random

stemmer = PorterStemmer()
from pathos.multiprocessing import ProcessingPool as Pool

from collections import defaultdict

import tensorflow as tf


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


class FeatureExtractionUtilities:
    bingnegs = []
    bingposs = []
    ade_list = []
    sentinegscores = {}
    sentiposscores = {}
    polarity_dict = {}
    topic_keys = {}
    word_clusters = defaultdict(list)
    neg_feature_vecs = np.zeros((100, 200), dtype="float32")
    neg_model = None
    pos_feature_vecs = np.zeros((100, 200), dtype="float32")
    pos_model = None
    goodwords = []
    badwords = []
    lesswords = []
    morewords = []

    @staticmethod
    def loadItems():
        # load bingliu items
        goodwords = []
        badwords = []
        lesswords = []
        morewords = []
        bingposs = []
        bingnegs = []
        ade_list = []
        topic_keys = {}
        word_clusters = defaultdict(list)
        infile = open('./sentimentanalysisresources/bingliunegs.txt')
        for line in infile:
            if not line[0] == ';':
                bingnegs.append(stemmer.stem(string.strip(line.decode('utf8', 'ignore').encode('ascii', 'ignore'))))
        infile = open('./sentimentanalysisresources/bingliuposs.txt')
        for line in infile:
            if not line[0] == ';':
                bingposs.append(stemmer.stem(string.strip(line.decode('utf8', 'ignore').encode('ascii', 'ignore'))))
        FeatureExtractionUtilities.bingnegs = bingnegs
        FeatureExtractionUtilities.bingposs = bingposs

        sentinegscores = {}
        sentiposscores = {}

        infile = open('sentimentanalysisresources/SentiWordNet_3.0.txt')
        next(infile)
        for line in infile:
            if not line[0] == '#' or not line[0] == ';':
                items = line.split('\t')
                pos = items[0]
                id_ = items[1]
                posscore = items[2]
                negscore = items[3]
                term = stemmer.stem(items[4][:items[4].index('#')].decode('utf8', 'ignore').encode('ascii', 'ignore'))
                # print term
                sentiposscores[(term, pos)] = posscore
                sentinegscores[(term, pos)] = negscore
        FeatureExtractionUtilities.sentinegscores = sentinegscores
        FeatureExtractionUtilities.sentiposscores = sentiposscores

        # loadclusters

        # load the subjectivity scores
        polarity_dict = {}
        infile = open('sentimentanalysisresources/subjectivity_score.tff')
        for line in infile:
            if not line[0] == ';':
                items = line.split()
                type = items[0][5:]
                word = stemmer.stem(items[2][6:])
                pos = items[3][5:]
                polaritystr = items[5][14:]
                # print type
                # print word
                # print type
                # print pos
                # print polaritystr
                multip = 0.0
                pol = 0.0
                if type == 'strongsubj':
                    multip = 1.0
                if type == 'weaksubj':
                    multip = 0.5
                if polaritystr == 'positive':
                    pol = 1.0
                if polaritystr == 'negative':
                    pol = -1.0
                if polaritystr == 'neutral':
                    pol = 0.0
                polval = multip * pol
                polarity_dict[(word, pos)] = polval

        infile = open('50mpaths2.txt')
        for line in infile:
            items = line.split()
            class_ = items[0]
            term = items[1]
            word_clusters[class_].append(term.decode('utf-8'))
        FeatureExtractionUtilities.word_clusters = word_clusters

        infile = open('ADR_lexicon.tsv')
        for line in infile:
            items = line.split('\t')
            # if len(items[1]) > 3:
            _name = string.strip(string.lower(items[1]))
            _cui = string.strip(items[0])
            ade_list.append(_name)  # (_cui,_name)
        FeatureExtractionUtilities.ade_list = ade_list

        max_weight = 0.0
        infile = open('TW_keys.txt')
        for line in infile:
            items = line.split()
            weight = float(items[1])
            if weight > max_weight:
                max_weight = weight
            for t in items[2:]:
                topic_keys[string.strip(t)] = weight

        for k in topic_keys.keys():
            topic_keys[k] = topic_keys[k] / max_weight
            # print max_weight
        FeatureExtractionUtilities.topic_keys = topic_keys

        # LOAD THE GOOD/BAD/MORE/LESS WORDS
        FeatureExtractionUtilities.loadgoodbadwords()

    @staticmethod
    def generateCentroidSimilarityScore(sent):
        # print sent
        terms = twokenize.tokenizeRawTweetText(sent)

        averagevec = np.zeros((300,), dtype="float32")
        for t in terms:
            try:
                averagevec = np.add(averagevec, FeatureExtractionUtilities.neg_model[t.lower()])
            except KeyError:
                pass
        try:
            averagevec = np.divide(averagevec, len(terms))
        except:
            pass

        sims = []
        for nfv in FeatureExtractionUtilities.neg_feature_vecs:
            sims.append(cosine(nfv, averagevec))
        averagepvec = np.zeros((300,), dtype="float32")
        for t in terms:
            try:
                averagepvec = np.add(averagepvec, FeatureExtractionUtilities.neg_model[t.lower()])
            except KeyError:
                pass
        if len(terms) > 0:
            averagevec = np.divide(averagepvec, len(terms))
        for nfv in FeatureExtractionUtilities.neg_feature_vecs:
            sims.append(cosine(nfv, averagepvec))

        # print sims
        return nan_to_num(sims)

    @staticmethod
    def makeFeatureVec(words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,), dtype="float32")
        #
        nwords = 0.
        #
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        #
        # Divide the result by the number of words to get the average
        if len(words) > 0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    @staticmethod
    def getAvgFeatureVecs(reviews, model, num_features):
        num_features = 300
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
        #
        # Loop through the reviews
        for k in reviews.keys():
            review = reviews[k]
            #
            # Print a status message every 1000th review
            if counter % 1000. == 0.:
                print "Review %d of %d" % (counter, len(reviews))
            #
            # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = FeatureExtractionUtilities.makeFeatureVec(review, model, num_features)
            #
            # Increment the counter
            counter = counter + 1.
        return reviewFeatureVecs

    @staticmethod
    def getclusterfeatures(sent):
        terms = twokenize.tokenizeRawTweetText(sent)
        # pos_tags = nltk.pos_tag(terms, 'universal')
        # terms = parsed_sent.split('\t')
        cluster_string = ''
        for t in terms:
            for k in FeatureExtractionUtilities.word_clusters.keys():
                if t in FeatureExtractionUtilities.word_clusters[k]:
                    cluster_string += ' clust_' + k + '_clust '
        return cluster_string

    @staticmethod
    def generateSemVec(processed_data, sequence_length, embed_size=10):
        semvec = []
        posScore = []
        negScore = []
        adeScore = []
        subjScore = []
        pposScore = []
        nnegScore = []
        moreGoodScore = []
        lessGoodScore = []
        moreBadScore = []
        lessBadScore = []
        clusterScore = []
        clusterScore2 = []
        wordLength=[]
        wordOrder = []

        # a,b=FeatureExtractionUtilities.getbingliuscores(processed_data,sequence_length)
        p = Pool(4)
        # for d in processed_data:
        for sentence_ade, sentence_cluster, sentence_cluster2, sentence_lbad, sentence_lgood, sentence_mbad, sentence_mgood, sentence_neg, sentence_nneg, sentence_pos, sentence_ppos, sentence_subj, sentence_vec,word_length,word_order in p.map(
                FeatureExtractionUtilities.generate_sentence_matrix, processed_data):
            arr = np.array(sentence_cluster)
            # assert arr.shape==(60,10)
            negScore.append(sentence_neg)
            posScore.append(sentence_pos)
            adeScore.append(sentence_ade)
            subjScore.append(sentence_subj)
            pposScore.append(sentence_ppos)
            nnegScore.append(sentence_nneg)
            moreGoodScore.append(sentence_mgood)
            moreBadScore.append(sentence_mbad)
            lessBadScore.append(sentence_lbad)
            lessGoodScore.append(sentence_lgood)
            clusterScore.append(arr)
            clusterScore2.append(np.array(sentence_cluster2))
            wordLength.append(word_length)
            wordOrder.append(word_order)

            semvec.append(sentence_vec)
        return semvec, negScore, posScore, adeScore, subjScore, pposScore, nnegScore, moreGoodScore, moreBadScore, lessBadScore, lessGoodScore, clusterScore, clusterScore2,wordLength,wordOrder

    @staticmethod
    def generate_Data_word2vec_matrix(processed_data):
        wordvec = []
        embedding_model = word2vec.Word2Vec.load_word2vec_format('word_embeddings.txt', binary=False)
        # a,b=FeatureExtractionUtilities.getbingliuscores(processed_data,sequence_length)
        p = Pool(4)

        doc_w2v_partial = partial(FeatureExtractionUtilities.generate_Document_word2vec, embedding_model=embedding_model)
        # for d in processed_data:
        for sentence_w2v in p.map(doc_w2v_partial,processed_data):
            wordvec.append(sentence_w2v)

        return wordvec

    @staticmethod
    def generate_Document_word2vec(d,embedding_model):
        sentence_vec = []

        sequence_length=32

        terms = twokenize.tokenizeRawTweetText(d.lower())
        min_len = min(sequence_length, len(terms))

        dict={}

        # pos_tags = nltk.pos_tag(terms, 'universal')
        # word_tokens = nltk.word_tokenize(d.lower())
        for i in range(0, min_len):
            if terms[i] in embedding_model:
                sentence_vec.append(np.array(embedding_model[terms[i]]))
            else:
                if(terms[i] in dict):
                    sentence_vec.append(dict[terms[i]])
                else:
                    dict[terms[i]]=np.array(np.random.uniform(-0.25, 0.25, embedding_model.vector_size))
                    sentence_vec.append(dict[terms[i]])
        # embedding_weights = [np.array([embedding_model[w] if w in embedding_model \
        #                                    else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
        #                                for w in vocabulary_inv])]
        # print embedding_model['medicine']
        diff = sequence_length - len(sentence_vec)
                # print diff
        if diff < 0:
           print diff
        for i in range(0, diff):
            sentence_vec.append(np.random.uniform(-0.25, 0.25, embedding_model.vector_size))
        return sentence_vec

    @staticmethod
    def generate_sentence_matrix(d):
        sentence_vec = []
        sentence_pos = []
        sentence_neg = []
        sentence_ade = []
        sentence_subj = []
        sentence_ppos = []
        sentence_nneg = []
        sentence_mgood = []
        sentence_lgood = []
        sentence_lbad = []
        sentence_mbad = []
        sentence_cluster = []
        sentence_cluster2 = []
        letters_count=[]
        word_order=[]

        added = False
        sequence_length = 32
        embed_size = 1
        zeros = -1 * np.ones(embed_size,
                             dtype=np.float64)  # np.random.uniform(low=0.0, high=1, size=embed_size) #np.zeros(embed_size)# random.uniform(low=0.0, high=1, size=embed_size)
        ones = np.ones(embed_size,
                       dtype=np.float64)  # np.random.uniform(low=0.0, high=1, size=1)#np.ones(embed_size, dtype=np.float32)
        two = np.ones(embed_size, dtype=np.float64)
        one = np.ones(1, dtype=np.float64)
        ades = '|'.join(FeatureExtractionUtilities.ade_list)

        terms = twokenize.tokenizeRawTweetText(d.lower())
        pos_tags = nltk.pos_tag(terms, 'universal')
        word_tokens = nltk.word_tokenize(d.lower())
        min_len = min(sequence_length, len(pos_tags))
        for i in range(0,min_len ):
            # print "pos_tags[i]"
            # print pos_tags[i]

            letters_count.append(len(pos_tags[i][0])*one)
            word_order.append(i*one)
            st = stemmer.stem(pos_tags[i][0])
            word_arr = np.array([])
            # print pos_tags[i][0]

            if st in FeatureExtractionUtilities.bingnegs:
                neg = 1 * np.ones(embed_size, dtype=np.float64)
                word_arr = np.concatenate((word_arr, neg), axis=0)
                sentence_pos.append(one)
                added = True
            else:
                word_arr = np.concatenate((word_arr, zeros), axis=0)
                sentence_pos.append(0 * one)

            if st in FeatureExtractionUtilities.bingposs:
                neg = 20 * np.ones(embed_size, dtype=np.float64)
                word_arr = np.concatenate((word_arr, neg), axis=0)
                sentence_neg.append(2 * one)
                added = True
            else:
                word_arr = np.concatenate((word_arr, zeros), axis=0)
                sentence_neg.append(0 * one)
            # rand = random.randint(1, 100)*ones
            # word_arr = np.concatenate((word_arr, rand), axis=0)

            # word_arr = np.concatenate((word_arr, rand), axis=0)
            # if re.search(ades, str(pos_tags[i][0])):
            #     word_arr = 3 * np.concatenate((word_arr, two), axis=0)
            #     sentence_ade.append(3 * one)
            # else:
            word_arr = np.concatenate((word_arr, zeros), axis=0)
            sentence_ade.append(0 * one)

            # word_arr = np.concatenate((word_arr, rand), axis=0)
            if FeatureExtractionUtilities.polarity_dict.has_key(st):
                i = terms.index(pos_tags[i])
                subjectivity_score = FeatureExtractionUtilities.polarity_dict[pos_tags[i][0]]
                score = 40 * subjectivity_score * ones
                word_arr = np.concatenate((word_arr, score), axis=0)
                sentence_subj.append(score)
                added = True
            else:
                sentence_subj.append(0 * one)
                word_arr = np.concatenate((word_arr, zeros), axis=0)

            if string.lower(str(pos_tags[i][1])) == 'adj':
                if FeatureExtractionUtilities.sentiposscores.has_key((st, 'a')):
                    posscore = float(
                        FeatureExtractionUtilities.sentiposscores[(st, 'a')])
                    score = 50 * posscore * np.ones(embed_size, dtype=np.float64)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    sentence_ppos.append(score)
                    added = True
                else:
                    word_arr = np.concatenate((word_arr, zeros), axis=0)
                    sentence_ppos.append(0 * one)

                if FeatureExtractionUtilities.sentinegscores.has_key((st, 'a')):
                    negscore = float(
                        FeatureExtractionUtilities.sentinegscores[(st, 'a')])
                    score = 6 * negscore * np.ones(embed_size, dtype=np.float64)
                    sentence_nneg.append(score)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    added = True
                else:
                    sentence_nneg.append(0 * one)
                    word_arr = np.concatenate((word_arr, zeros), axis=0)

            elif string.lower(str(pos_tags[i][1])) == 'verb':
                if FeatureExtractionUtilities.sentiposscores.has_key((st, 'v')):
                    posscore = float(
                        FeatureExtractionUtilities.sentiposscores[(st, 'v')])
                    score = 50 * posscore * np.ones(embed_size, dtype=np.float64)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    sentence_ppos.append(score)
                    added = True
                else:
                    word_arr = np.concatenate((word_arr, zeros), axis=0)
                    sentence_ppos.append(0 * one)
                if FeatureExtractionUtilities.sentinegscores.has_key((st, 'v')):
                    negscore = float(
                        FeatureExtractionUtilities.sentinegscores[(st, 'v')])
                    score = 6 * negscore * np.ones(embed_size, dtype=np.float64)
                    sentence_nneg.append(score)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    added = True
                else:
                    sentence_nneg.append(0 * one)
                    word_arr = np.concatenate((word_arr, zeros), axis=0)
            elif string.lower(str(pos_tags[i][1])) == 'noun':
                if FeatureExtractionUtilities.sentiposscores.has_key((st, 'n')):
                    posscore = float(
                        FeatureExtractionUtilities.sentiposscores[(st, 'n')])
                    score = 50 * posscore * np.ones(embed_size, dtype=np.float64)
                    sentence_ppos.append(score)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    added = True
                else:
                    word_arr = np.concatenate((word_arr, zeros), axis=0)
                    sentence_ppos.append(0 * one)
                if FeatureExtractionUtilities.sentinegscores.has_key((st, 'n')):
                    negscore = float(
                        FeatureExtractionUtilities.sentinegscores[(st, 'n')])
                    score = 6 * negscore * np.ones(embed_size, dtype=np.float64)
                    sentence_nneg.append(score)
                    word_arr = np.concatenate((word_arr, score), axis=0)
                    added = True
                else:
                    sentence_nneg.append(0 * one)
                    word_arr = np.concatenate((word_arr, zeros), axis=0)
            else:
                sentence_ppos.append(0 * one)
                sentence_nneg.append(0 * one)
                word_arr = np.concatenate((word_arr, zeros), axis=0)
                word_arr = np.concatenate((word_arr, zeros), axis=0)

            moreGood = 0
            moreBad = 0
            lessGood = 0
            lessBad = 0

            for k in range(i, len(word_tokens)):
                if st in FeatureExtractionUtilities.morewords:

                    minboundary = max(i - 4, 0)
                    maxboundary = min(i + 4, len(pos_tags) - 1)
                    j = minboundary
                    while j <= maxboundary:
                        t = stemmer.stem(pos_tags[j][0])
                        if t in FeatureExtractionUtilities.goodwords:
                            moreGood = 70
                        elif t in FeatureExtractionUtilities.badwords:
                            moreBad = 8
                        j += 1
                if st in FeatureExtractionUtilities.lesswords:
                    minboundary = max(i - 4, 0)
                    maxboundary = min(i + 4, len(pos_tags) - 1)
                    j = minboundary
                    while j <= maxboundary:
                        t = stemmer.stem(pos_tags[j][0])
                        if t in FeatureExtractionUtilities.goodwords:
                            lessGood = 90

                        elif t in FeatureExtractionUtilities.badwords:
                            lessBad = 10

                        j += 1
            sentence_mgood.append(moreGood * one)
            sentence_mbad.append(moreBad * one)
            sentence_lgood.append(lessGood * one)
            sentence_lbad.append(lessBad * one)

            clusterAdded = False
            # for k in FeatureExtractionUtilities.word_clusters.keys():
            #     if st in FeatureExtractionUtilities.word_clusters[k]:
            #         clusterAdded = True
            #         kdiff = 20 - len(k)
            #         clust = np.concatenate((np.zeros(kdiff, dtype=np.float64), np.array(list(k), dtype=np.float64)),
            #                                axis=0)
            #         wclass = np.array(clust[:10],
            #                           dtype=np.float64)  # np.concatenate((np.zeros(kdiff, dtype=np.float64), np.array(list(k), dtype=np.float64)), axis=0)
            #         wclass2 = np.array(clust[10:], dtype=np.float64)
            #         # if(kdiff>0):
            #         #  wclass=np.concatenate((np.zeros(kdiff, dtype=np.float64), wclass), axis=0)
            #
            #         sentence_cluster.append(wclass)
            #         sentence_cluster2.append(wclass2)
            if clusterAdded == False:
                sentence_cluster.append(np.zeros(10, dtype=np.float64))
                sentence_cluster2.append(np.zeros(10, dtype=np.float64))
            word_arr = np.concatenate((word_arr, [moreGood, moreBad, lessGood, lessBad]), axis=0)

            sentence_vec.append(word_arr)
        diff = sequence_length - len(sentence_vec)
        # print diff
        if diff < 0:
            print diff
        for i in range(0, diff):
            z = np.concatenate((zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros),
                               axis=0)  # ,zeros,zeros,zeros,zeros
            letters_count.append(0* one)
            sentence_vec.append(z)
            sentence_neg.append(0 * one)
            sentence_pos.append(0 * one)
            sentence_ade.append(0 * one)
            sentence_subj.append(0 * one)
            sentence_ppos.append(0 * one)
            sentence_nneg.append(0 * one)
            sentence_lgood.append(0 * one)
            sentence_mbad.append(0 * one)
            sentence_lbad.append(0 * one)
            sentence_mgood.append(0 * one)
            sentence_cluster.append(np.zeros(10, dtype=np.float64))
            sentence_cluster2.append(np.zeros(10, dtype=np.float64))
            word_order.append(0 * one)
        return sentence_ade, sentence_cluster, sentence_cluster2, sentence_lbad, sentence_lgood, sentence_mbad, sentence_mgood, sentence_neg, sentence_nneg, sentence_pos, sentence_ppos, sentence_subj, sentence_vec,letters_count,word_order

    @staticmethod
    def getbingliuscores(processed_data, sequence_length):
        bingposcount = 0.0
        bingnegcount = 0.0
        bposcounts = []
        bnegcounts = []
        zeros = np.random.uniform(low=0.0, high=1, size=6)
        ones = 100 * np.ones(6, dtype=np.float32)
        minusones = 120 * np.ones(6, dtype=np.float32)

        for d in processed_data:
            bingposcount = 0.0
            bingnegcount = 0.0
            itmbingpos = []
            itmbingneg = []

            items = d.split()
            for i in items:
                if i in FeatureExtractionUtilities.bingnegs:
                    # bingnegcount =1
                    itmbingpos.append(ones)
                else:
                    itmbingpos.append(zeros)

                if i in FeatureExtractionUtilities.bingposs:
                    itmbingneg.append(minusones)
                else:
                    itmbingneg.append(zeros)
            diff = sequence_length - len(itmbingpos)
            for i in range(0, diff):
                itmbingpos.append(zeros)
                itmbingneg.append(zeros)
            # a = np.zeros((diff), dtype=np.float64).tolist()
            # pos = np.concatenate((np.array(itmbingpos), a), axis=0)
            # neg = np.concatenate((np.array(itmbingneg), a), axis=0)
            bposcounts.append(itmbingpos)
            bnegcounts.append(itmbingneg)
        # diff = sequence_length - len(bposcounts[1])
        # a = np.zeros((len(processed_data), diff), dtype=np.float64).tolist()
        # print len(a)==len(processed_data)
        # pos = np.concatenate((np.array(bposcounts), a), axis=0)
        # neg = np.concatenate((np.array(bnegcounts), a), axis=0)
        # print bnegcounts
        # print bposcounts
        return bposcounts, bnegcounts

    @staticmethod
    def getsentiwordscores(processed_data, sequence_length):
        negscore = 0.0
        posscore = 0.0
        negscores = []
        posscores = []
        zeros = np.random.uniform(low=0.0, high=1, size=6)
        for d in processed_data:
            negscore = 0.0
            posscore = 0.0
            sentencenegscore = []
            sentenceposscore = []

            terms = twokenize.tokenizeRawTweetText(d)
            pos_tags = nltk.pos_tag(terms, 'universal')
            for i in range(0, len(pos_tags)):
                try:
                    if string.lower(str(pos_tags[i][1])) == 'adj':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'a')):
                            posscore = float(
                                FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'a')])
                            score = 100 * posscore * np.ones(6, dtype=np.float64)
                            sentenceposscore.append(score)
                        else:
                            sentenceposscore.append(zeros)

                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'a')):
                            negscore = float(
                                FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'a')])
                            score = 100 * negscore * np.ones(6, dtype=np.float64)
                            sentencenegscore.append(score)
                        else:
                            sentencenegscore.append(zeros)

                    elif string.lower(str(pos_tags[i][1])) == 'verb':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'v')):
                            posscore = float(
                                FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'v')])
                            score = 100 * posscore * np.ones(6, dtype=np.float64)
                            sentenceposscore.append(score)
                        else:
                            sentenceposscore.append(zeros)
                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'v')):
                            negscore = float(
                                FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'v')])
                            score = 100 * negscore * np.ones(6, dtype=np.float64)
                            sentencenegscore.append(score)
                        else:
                            sentencenegscore.append(zeros)
                    elif string.lower(str(pos_tags[i][1])) == 'noun':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'n')):
                            posscore = float(
                                FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'n')])
                            score = 100 * posscore * np.ones(6, dtype=np.float64)
                            sentenceposscore.append(score)
                        else:
                            sentenceposscore.append(zeros)
                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'n')):
                            negscore = float(
                                FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'n')])
                            score = 100 * negscore * np.ones(6, dtype=np.float64)
                            sentencenegscore.append(score)
                        else:
                            sentencenegscore.append(zeros)
                    else:
                        sentencenegscore.append(zeros)
                        sentenceposscore.append(zeros)
                except Exception:
                    # sentencenegscore.append(zeros)
                    # sentenceposscore.append(zeros)
                    pass
            # diff = sequence_length - len(sentencenegscore)
            # a = np.zeros((diff), dtype=np.float64).tolist()
            # pos = np.concatenate((np.array(sentenceposscore), a), axis=0)
            # neg = np.concatenate((np.array(sentencenegscore), a), axis=0)
            diff = sequence_length - len(terms)
            for i in range(0, diff):
                sentencenegscore.append(zeros)
                sentenceposscore.append(zeros)
            negscores.append(sentencenegscore)
            posscores.append(sentenceposscore)

        # print negscores
        # print posscores
        return negscores, posscores

    @staticmethod
    def getsubjectivityscores(processed_data, sequence_length):
        subjectivity_scores = []
        for d in processed_data:
            subjectivity_score = 0.0
            subjectivity_score_lst = []
            # score = np.zeros((sequence_length), dtype=np.float64).tolist()
            zeros = np.random.uniform(low=0.0, high=1, size=6)
            ones = np.ones(6, dtype=np.float32)
            terms = twokenize.tokenizeRawTweetText(d)
            pos_tags = nltk.pos_tag(terms, 'universal')
            for i in range(0, len(pos_tags)):
                try:
                    if FeatureExtractionUtilities.polarity_dict.has_key(pos_tags[i]):
                        i = terms.index(pos_tags[i])
                        subjectivity_score = FeatureExtractionUtilities.polarity_dict[pos_tags[i]]
                        score = 100 * subjectivity_score * ones
                        subjectivity_score_lst.append(score)
                    else:
                        subjectivity_score_lst.append(zeros)
                except Exception:
                    subjectivity_score_lst.append(zeros)
                    pass
            # subjectivity_score = subjectivity_score/len(terms)
            # diff = sequence_length - len(subjectivity_score_lst)
            diff = sequence_length - len(subjectivity_score_lst)
            for i in range(0, diff):
                subjectivity_score_lst.append(zeros)

            # score = np.concatenate((np.array(subjectivity_score_lst), a), axis=0)
            subjectivity_scores.append(subjectivity_score_lst)

        return subjectivity_scores

    @staticmethod
    def getlexiconfeatures(processed_data, sequence_length):
        lexicon_features = []
        ades = '|'.join(map(re.escape, FeatureExtractionUtilities.ade_list))

        for d in processed_data:
            ade_presence = 0
            ade_count = 0.0
            # to make sure that concepts with the same cui are not searched multiple times
            score = np.zeros((sequence_length), dtype=np.float64).tolist()
            addedcuilist = []
            # words=string.lower(d)
            # sentence = nltk.word_tokenize(words)
            # ngrams=[]
            # # ngrams.append(find_ngrams(sentence, 1))
            # # ngrams.append(find_ngrams(sentence, 2))
            # # ngrams.append(find_ngrams(sentence, 3))
            # # ngrams.append(find_ngrams(sentence, 5))
            # # sentence.append(ngrams)
            # # for (cui, ade) in FeatureExtractionUtilities.ade_list:
            # for i in range(0, len(sentence)-1):
            #     if re.search(ades,sentence[i]):
            #             score[i] = 1.0
            # ade_presence = 1.0

            # ade_count = ade_count/len(sentence.split())
            # lexicon_features.append([ade_presence,ade_count])
            # print 'score==>'+str(score)
            lexicon_features.append(score)
        # print lexicon_features
        # print lexicon_features
        return lexicon_features

    @staticmethod
    def gettopicscores(processed_data):
        topic_features = []
        topic_texts = []
        for d in processed_data:
            weighted_score = 0.0
            # topic_presence = 0
            topic_terms = ''
            for k in FeatureExtractionUtilities.topic_keys.keys():
                if k in d.split():
                    topic_presence = 1
                    topic_terms += 'top_' + k + '_top '
                    weighted_score += FeatureExtractionUtilities.topic_keys[k]
            topic_features.append([weighted_score])
            topic_texts.append(topic_terms)
        return topic_texts, topic_features
        # return topic_terms, weighted_score

    @staticmethod
    def getsentimentfeatures(processed_data, sequence_length):
        # print 'processed_data'+str(processed_data)
        negcounts, poscounts = FeatureExtractionUtilities.getbingliuscores(processed_data, sequence_length)
        negscores, posscores = FeatureExtractionUtilities.getsentiwordscores(processed_data, sequence_length)
        subjectivity_scores = FeatureExtractionUtilities.getsubjectivityscores(processed_data, sequence_length)
        # print 'negcounts,poscounts'+str(negcounts)
        # print'poscounts'+str(poscounts)
        # print 'negscores,posscores'+str(negscores)
        # print ' '+str(posscores)
        # trans =np.column_stack((poscounts,negcounts,negscores,posscores,subjectivity_scores))
        # print(trans)
        # X=tf.concat(2, [negcounts, poscounts, negscores, posscores, subjectivity_scores])


        # print 'subjectivity_scores'+str(subjectivity_scores)
        # features = map(list.__add__,poscounts,negcounts)
        # features2 = map(list.__add__,posscores,negscores)
        # features = map(list.__add__,features,features2)
        # features = map(list.__add__,features,subjectivity_scores)
        # print 'features'+str(features)
        return poscounts, negcounts, negscores, posscores, subjectivity_scores

    @staticmethod
    def getstructuralfeatures(processed_data):
        lens = FeatureExtractionUtilities.getreviewlengths(processed_data)
        numsents = FeatureExtractionUtilities.getnumsentences(processed_data)
        avelengths = FeatureExtractionUtilities.getaveragesentlengths(processed_data)

        features = map(list.__add__, lens, avelengths)
        features = map(list.__add__, features, numsents)
        return features

    @staticmethod
    def getreviewlengths(processed_data):
        lengths = []
        for d in processed_data:
            items = d.split()
            lengths.append([len(items)])

        return lengths

    @staticmethod
    def getnumsentences(processed_data):
        numsents = []
        for d in processed_data:
            items = nltk.sent_tokenize(d)
            numsents.append([len(items)])
        return numsents

    @staticmethod
    def getaveragesentlengths(processed_data):
        avelengths = []
        for d in processed_data:
            items = nltk.sent_tokenize(d)
            words = d.split()
            numsents = len(items)
            numwords = len(words)
            avelengths.append([numwords / (numsents + 0.0)])
        return avelengths

    @staticmethod
    def getSynsetString(sent, negations):
        terms = twokenize.tokenizeRawTweetText(sent)
        pos_tags = nltk.pos_tag(terms, 'universal')

        # terms = parsed_sent.split('\t')
        sent_terms = []
        # now terms[0] will contain the text and terms[1] will contain the POS (space separated)
        # sentence_tokens = terms[0].split()
        # pos_tags = terms[1].split()
        for i in range(0, len(pos_tags)):

            if string.lower(str(pos_tags[i][1])) == 'adj':
                synsets = wn.synsets(string.lower(pos_tags[i][0]), pos=wn.ADJ)
                for syn in synsets:
                    lemmas = [string.lower(lemma) for lemma in syn.lemma_names()]
                    sent_terms += lemmas

            if string.lower(pos_tags[i][1]) == 'verb':
                synsets = wn.synsets(string.lower(pos_tags[i][0]), pos=wn.VERB)
                for syn in synsets:
                    lemmas = [string.lower(lemma) for lemma in syn.lemma_names()]
                    sent_terms += lemmas

            if string.lower(pos_tags[i][1]) == 'noun':

                synsets = wn.synsets(string.lower(pos_tags[i][0]), pos=wn.NOUN)
                for syn in synsets:
                    lemmas = [string.lower(lemma) for lemma in syn.lemma_names()]

                    sent_terms += lemmas
        sent_terms = list(set(sent_terms))
        # print sent_terms
        senttermsstring = ''

        for term in sent_terms:
            senttermsstring += ' ' + 'syn_' + stemmer.stem(term) + '_syn'
        # print senttermsstring
        return senttermsstring

    @staticmethod
    def detectModals(senttokens, modals):
        for word in senttokens:
            for item in modals:
                if cmp(string.lower(item), string.lower(word)) == 0:
                    return 1
        return 0

    @staticmethod
    def loadModals():
        modals = []
        infile = open('./polaritycues/modals.txt')
        for line in infile:
            modals.append(string.strip(line))
        return modals

    @staticmethod
    def loadGoodWords():
        goodwords = []
        infile = open('./polaritycues/new_good_words.txt')
        for line in infile:
            goodwords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.goodwords = goodwords

    @staticmethod
    def loadBadWords():
        badwords = []
        infile = open('./polaritycues/new_bad_words.txt')
        for line in infile:
            badwords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.badwords = badwords

    @staticmethod
    def loadMoreWords():
        morewords = []
        infile = open('./polaritycues/more_words.txt')
        for line in infile:
            morewords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.morewords = morewords

    @staticmethod
    def loadLessWords():
        lesswords = []
        infile = open('./polaritycues/less_words.txt')
        for line in infile:
            lesswords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.lesswords = lesswords

    @staticmethod
    def loadgoodbadwords():
        FeatureExtractionUtilities.loadGoodWords()
        FeatureExtractionUtilities.loadBadWords()
        FeatureExtractionUtilities.loadMoreWords()
        FeatureExtractionUtilities.loadLessWords()

    '''
        Given a sentence, addsmore/less..good/bad features as proposed by niu et al.
        once a more/less word is found, a 4 word window is inspected on either side to detect
         the presence of good/bad words.
         returns a binary vector of the form: [moregood,morebad,lessgood,lessbad]
    '''

    @staticmethod
    def goodbadFeatures(processed_data):
        goodbadfeatures = []
        for pd in processed_data:

            moreGood = 0
            moreBad = 0
            lessGood = 0
            lessBad = 0
            sentence = string.lower(pd)
            word_tokens = nltk.word_tokenize(sentence)
            for i in range(0, len(word_tokens)):
                stemmedword = word_tokens[i]
                if stemmedword in FeatureExtractionUtilities.morewords:

                    minboundary = max(i - 4, 0)
                    maxboundary = min(i + 4, len(word_tokens) - 1)
                    j = minboundary
                    while j <= maxboundary:
                        if word_tokens[j] in FeatureExtractionUtilities.goodwords:
                            moreGood = 1
                        elif word_tokens[j] in FeatureExtractionUtilities.badwords:
                            moreBad = 1
                        j += 1
                if stemmedword in FeatureExtractionUtilities.lesswords:
                    minboundary = max(i - 4, 0)
                    maxboundary = min(i + 4, len(word_tokens) - 1)
                    j = minboundary
                    while j <= maxboundary:
                        if word_tokens[j] in FeatureExtractionUtilities.goodwords:
                            lessGood = 1
                        elif word_tokens[j] in FeatureExtractionUtilities.badwords:
                            lessBad = 1
                        j += 1
            goodbadfeatures.append([moreGood, moreBad, lessGood, lessBad])
        return goodbadfeatures