from collections import Counter
import nltk
import pandas as pd

import data_helpers
from TwitterCleanuper import TwitterCleanuper
from data_helpers import clean_str, clean_tweet, load_semeval_and_y_labels
from emoticons import EmoticonDetector
import re as regex
import csv
from collections import OrderedDict, defaultdict, Counter
import numpy
from featureextractionmodules import twokenize
from nltk.tokenize import TweetTokenizer
from pathos.multiprocessing import ProcessingPool as Pool

def generate_emotion_features(text):
    terms = twokenize.tokenizeRawTweetText(text)
    tags = nltk.pos_tag(terms, 'universal')
    # print tags

    anger_list = numpy.zeros(32)
    anticipation_list = numpy.zeros(32)
    disgust_list = numpy.zeros(32)
    fear_list = numpy.zeros(32)
    joy_list = numpy.zeros(32)
    negative_list = numpy.zeros(32)
    positive_list = numpy.zeros(32)
    sadness_list = numpy.zeros(32)
    surprise_list = numpy.zeros(32)
    trust_list = numpy.zeros(32)

    anger_cnt = 0
    anticipation_cnt = 0
    disgust_cnt = 0
    fear_cnt = 0
    joy_cnt = 0
    negative_cnt = 0
    positive_cnt = 0
    sadness_cnt = 0
    surprise_cnt = 0
    trust_cnt = 0

    min_len = min(32, len(tags))
    # found_emotions = 0
    for i in range(0, min_len):
        print tags[i][0]
        print wordList[tags[i][0]]
        if 'anger' in wordList[tags[i][0]]:
            anger_list[i] = 13
            anger_cnt = anger_cnt + 1
        if 'anticipation' in wordList[tags[i][0]]:
            anticipation_list[i] = 14
            anticipation_cnt = anticipation_cnt + 1
        if 'disgust' in wordList[tags[i][0]]:
            disgust_list[i] = 15
            disgust_cnt = disgust_cnt + 1
        if 'fear' in wordList[tags[i][0]]:
            fear_list[i] = 16
            fear_cnt = fear_cnt + 1
        if 'joy' in wordList[tags[i][0]]:
            joy_list[i] = 17
            joy_cnt = joy_cnt + 1
        if 'negative' in wordList[tags[i][0]]:
            negative_list[i] = 18
            negative_cnt = negative_cnt + 1
        if 'positive' in wordList[tags[i][0]]:
            positive_list[i] = 19
            positive_cnt = positive_cnt + 1
        if 'sadness' in wordList[tags[i][0]]:
            sadness_list[i] = 20
            sadness_cnt = sadness_cnt + 1
        if 'surprise' in wordList[tags[i][0]]:
            surprise_list[i] = 21
            surprise_cnt = surprise_cnt + 1
        if 'trust' in wordList[tags[i][0]]:
            trust_list[i] = 22
            trust_cnt = trust_cnt + 1

    if anger_cnt > 1:
        anger_list = anger_list * anger_cnt
    if anticipation_cnt > 1:
        anticipation_list = anticipation_list * anticipation_cnt
    if disgust_cnt > 1:
        disgust_list = disgust_list * disgust_cnt
    if fear_cnt > 1:
        fear_list = fear_list * fear_cnt
    if joy_cnt > 1:
        joy_list = joy_list * joy_cnt
    if negative_cnt > 1:
        negative_list = negative_list * negative_cnt
    if positive_cnt > 1:
        positive_list = positive_list * positive_cnt
    if sadness_cnt > 1:
        sadness_list = sadness_list * sadness_cnt
    if surprise_cnt > 1:
        surprise_list = surprise_list * surprise_cnt
    if trust_cnt > 1:
        trust_list = trust_list * trust_cnt

    return (anger_list,anticipation_list,disgust_list,fear_list,joy_list,negative_list,positive_list,sadness_list,surprise_list,trust_list)
wordList = defaultdict(list)
emotionList = defaultdict(list)
with open('../lexicons/NRC-Emotion-Lexicon-v0.92.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if int(present) == 1:
            #print(word)
            wordList[word].append(emotion)
            emotionList[emotion].append(word)

emoticons = {}

# emoticon_file="../data/emoticons.txt"
# from pathlib import Path
# content = open(emoticon_file, "r").read()
# positive = True
# for line in content.split("\n"):
#     if "positive" in line.lower():
#         positive = True
#         continue
#     elif "negative" in line.lower():
#         positive = False
#         continue
#
#     emoticons[line] = positive

tt = TweetTokenizer()
def generate_emotion_count(string, tokenizer):
    emoCount = Counter()
    for token in tt.tokenize(string):
        token = token.lower()
        emoCount += Counter(wordList[token])
    return emoCount

def is_positive(self, emoticon):
    if emoticon in self.emoticons:
        return self.emoticons[emoticon]
    return False

def is_emoticon(to_check):
    return to_check in emoticons

d="@Ayerad no, well  i hope not. He could ha hasnt been at school fer a wile  but @koast08 doesnt believe he had cancer"

# clean= clean_str(d)
# print clean_tweet(d)
# clean=clean_tweet(d)
# print clean
x_text, Y = data_helpers.load_data_and_y_labels("../data/MR/rt-polarity.pos",
                                                    "../data/MR/rt-polarity.neg")

# cleanuper=TwitterCleanuper()
# for cleanup_method in cleanuper.iterate():
#         # if cleanup_method.__name__ != "remove_na":
#         x_text = cleanup_method(x_text)

# print d

# terms = twokenize.tokenizeRawTweetText(clean)
# print tt.tokenize(d)



# sentence_arr = numpy.zeros(32)
# print wordList['good']
#
#
# if wordList['good']:
#     print 'anticipation' in wordList['good']
#
# emotions=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
#        'positive', 'sadness', 'surprise', 'trust']


anger_feature_list = []
anticipation_feature_list = []
disgust_feature_list = []
fear_feature_list = []
joy_feature_list = []
negative_feature_list = []
positive_feature_list = []
sadness_feature_list = []
surprise_feature_list = []
trust_feature_list = []
p = Pool(4)
for (anger_list,anticipation_list,disgust_list,fear_list,joy_list,negative_list,positive_list,sadness_list,surprise_list,trust_list) in p.map(generate_emotion_features, x_text):
    anger_feature_list.append(anger_list)
    anticipation_feature_list.append(anticipation_list)
    disgust_feature_list.append(disgust_list)
    fear_feature_list.append(fear_list)
    joy_feature_list.append(joy_list)
    negative_feature_list.append(negative_list)
    positive_feature_list.append(positive_list)
    sadness_feature_list.append(sadness_list)
    surprise_feature_list.append(surprise_list)
    trust_feature_list.append(trust_list)


anger_feature_list = numpy.expand_dims(anger_feature_list, axis=2)
anticipation_feature_list = numpy.expand_dims(anticipation_feature_list, axis=2)
disgust_feature_list = numpy.expand_dims(disgust_feature_list, axis=2)
fear_feature_list = numpy.expand_dims(fear_feature_list, axis=2)
joy_feature_list = numpy.expand_dims(joy_feature_list, axis=2)
negative_feature_list = numpy.expand_dims(negative_feature_list, axis=2)
positive_feature_list = numpy.expand_dims(positive_feature_list, axis=2)
sadness_feature_list = numpy.expand_dims(sadness_feature_list, axis=2)
surprise_feature_list = numpy.expand_dims(surprise_feature_list, axis=2)
trust_feature_list = numpy.expand_dims(trust_feature_list, axis=2)

data_set="MR"
numpy.save("../dump/" + data_set + "/anger_feature_list", anger_feature_list)
numpy.save("../dump/" + data_set + "/anticipation_feature_list", anticipation_feature_list)
numpy.save("../dump/" + data_set + "/disgust_feature_list", disgust_feature_list)
numpy.save("../dump/" + data_set + "/fear_feature_list", fear_feature_list)
numpy.save("../dump/" + data_set + "/joy_feature_list", joy_feature_list)
numpy.save("../dump/" + data_set + "/negative_feature_list", negative_feature_list)
numpy.save("../dump/" + data_set + "/positive_feature_list", positive_feature_list)
numpy.save("../dump/" + data_set + "/sadness_feature_list", sadness_feature_list)
numpy.save("../dump/" + data_set + "/surprise_feature_list", surprise_feature_list)
numpy.save("../dump/" + data_set + "/trust_feature_list", trust_feature_list)