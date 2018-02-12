import string
from collections import Counter
import nltk
import pandas as pd

from TwitterCleanuper import TwitterCleanuper
from data_helpers import clean_str, clean_tweet, load_semeval_and_y_labels
import re as regex
import csv
from collections import OrderedDict, defaultdict, Counter
import numpy
from featureextractionmodules import twokenize
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
from nltk.tokenize import TweetTokenizer
from pathos.multiprocessing import ProcessingPool as Pool
from nltk.stem.porter import *
import pandas as pd
import io


stemmer = PorterStemmer()
def generate_senti_features(text):
    terms = twokenize.tokenizeRawTweetText(text)
    pos_tags = nltk.pos_tag(terms, 'universal')
    # print tags

    sentence_pos = numpy.zeros(32)
    sentence_neg = numpy.zeros(32)

    min_len = min(32, len(pos_tags))

    for i in range(0, min_len):
        st = stemmer.stem(pos_tags[i][0])
        # print pos_tags[i][0]

        if string.lower(str(pos_tags[i][1])) == 'adj':
            if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'a')):
                posscore = float(
                    FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'a')])
                score = 200 * posscore
                sentence_pos[i]=score


            if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'a')):
                negscore = float(
                    FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'a')])
                score = 100 * negscore
                sentence_neg[i]=score

        elif string.lower(str(pos_tags[i][1])) == 'verb':
            if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'v')):
                posscore = float(
                    FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'v')])
                score = 200 * posscore
                sentence_pos[i] = score
            if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'v')):
                negscore = float(
                    FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'v')])
                score = 100 * negscore
                sentence_neg[i] = score

        elif string.lower(str(pos_tags[i][1])) == 'noun':
            if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])), 'n')):
                posscore = float(
                    FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])), 'n')])
                score = 200 * posscore
                sentence_pos[i] = score

            if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])), 'n')):
                negscore = float(
                    FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])), 'n')])
                score = 100 * negscore
                sentence_neg[i] = score
    return (sentence_pos,sentence_neg)

FeatureExtractionUtilities.loadItems()

x_text, Y = load_semeval_and_y_labels("data/SemEval2015-task10-test-B-input.txt")


pos_feature_list = []
neg_feature_list = []


p = Pool(4)
for (pos_list,neg_list) in p.map(generate_senti_features, x_text):
    pos_feature_list.append(pos_list)
    neg_feature_list.append(neg_list)



pos_feature_list = numpy.expand_dims(pos_feature_list, axis=2)
neg_feature_list = numpy.expand_dims(neg_feature_list, axis=2)



data_set="SemEval2015"
numpy.save("dump/" + data_set + "Test/pposScore", pos_feature_list)
numpy.save("dump/" + data_set + "Test/nnegScore", neg_feature_list)
