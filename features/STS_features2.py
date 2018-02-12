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

import pandas as pd
import io

def generate_emotion_features(text):
    terms = twokenize.tokenizeRawTweetText(text)
    tags = nltk.pos_tag(terms, 'universal')
    # print tags

    valence_list = numpy.zeros(32)
    arousal_list = numpy.zeros(32)
    dominance_list = numpy.zeros(32)

    min_len = min(32, len(tags))

    for i in range(0, min_len):
        if valenceList[tags[i][0]]:
            valence_list[i]= valenceList[tags[i][0]][0]
            arousal_list[i]= arousalList[tags[i][0]][0]
            dominance_list[i]= dominanceList[tags[i][0]][0]




    return (valence_list,arousal_list,dominance_list)
valenceList = defaultdict(list)
arousalList = defaultdict(list)
dominanceList=defaultdict(list)
with open('../lexicons/Ratings_Warriner_et_al.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=',')

    for line in reader:
        w = line["Word"]
        valenceList[w].append(line["V.Mean.Sum"])
        arousalList[w].append(line["A.Mean.Sum"])
        dominanceList[w].append(line["D.Mean.Sum"])



d="@Ayerad no, well  i hope not. He could ha hasnt been at school fer a wile  but @koast08 doesnt believe he had cancer"
a,b,c=generate_emotion_features(d)

x_text, Y = data_helpers.load_data_and_y_labels("../data/MR/rt-polarity.pos",
                                                    "../data/MR/rt-polarity.neg")


valence_feature_list = []
arousal_feature_list = []
dominance_feature_list = []

p = Pool(4)
for (valence_list,arousal_list,dominance_list) in p.map(generate_emotion_features, x_text):
    valence_feature_list.append(valence_list)
    arousal_feature_list.append(arousal_list)
    dominance_feature_list.append(dominance_list)


valence_feature_list = numpy.expand_dims(valence_feature_list, axis=2)
arousal_feature_list = numpy.expand_dims(arousal_feature_list, axis=2)
dominance_feature_list = numpy.expand_dims(dominance_feature_list, axis=2)


data_set="MR"
numpy.save("../dump/" + data_set + "/valence_feature_list", valence_feature_list)
numpy.save("../dump/" + data_set + "/arousal_feature_list", arousal_feature_list)
numpy.save("../dump/" + data_set + "/dominance_feature_list", dominance_feature_list)
