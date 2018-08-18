import cupy

from data_helpers import clean_str, load_data_and_y_labels
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
from featureextractionmodules import twokenize
import nltk,re,string
from pathos.multiprocessing import ProcessingPool as Pool
#Add these lines:
import nltk
from nltk.corpus import wordnet as WN
from nltk.corpus import stopwords

from spellchecker.spellcheck import correct

stop_words_en = set(stopwords.words('english'))

def generate_ade_feature(text):
    terms = twokenize.tokenizeRawTweetText(text)
    tags = nltk.pos_tag(terms, 'universal')
    sentence_arr = cupy.zeros(32)
    ade_count = 0
    min_len = min(32, len(tags))
    for i in range(0, min_len):
        found_adr = False
        sentence_list = find_words(tags, i)
        for sentence in sentence_list:
            if find_adr(sentence):
                # print "sentence {}".format(sentence)
                ln = len(nltk.word_tokenize(sentence))
                ade_count = ade_count + 1
                sentence_arr[i:i + ln] = 10
                found_adr = True
        if tags[i][0] not in stop_words_en  and  not found_adr and (tags[i][1] == 'ADJ' or tags[i][1] == 'ADV'or tags[i][1] == 'NOUN'or tags[i][1] == 'VERB'):
            if re.search(r'\b{0}\b'.format(tags[i][0]),ades):
                ade_count = ade_count + 1
                sentence_arr[i] = 1
    # print ade_count
    sentence_arr = sentence_arr * ade_count
    sentence_ade.append(sentence_arr)
    # print sentence_arr
    return sentence_arr

def find_adr(sentence):
    to_search='|'+sentence+'|'
    if to_search in ades or sentence == 'crippled' or sentence=='infection vascular':
       return True
    return False

def tokens(sent):
        return nltk.word_tokenize(sent)

def SpellChecker(line):
    for i in tokens(line):
        strip = i.rstrip()
        if not WN.synsets(strip):
            if strip not in stop_words_en:    # <--- Check whether it's in stopword list
                print("Wrong spellings : " + i)
                cor=correct(i)
                print cor
                line=line.replace(i,cor)
    return line


def removePunct(str):
        return  "".join(c for c in str if c not in ('!','.',':',','))

def findWholeWord(w):
    return re.compile(r'\b{0}\b'.format(w), flags=re.IGNORECASE).search


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def find_words(input_list,sindex):
    ln=len(input_list)
    sen_list=[]
    end = min(sindex + 4, ln)
    lst=[]

    for i in range(sindex + 1, end + 1):
        lst.append(' '.join([tup[0] for tup in input_list[sindex:i]]))
    # print lst
    for str in lst:
        if str not in sen_list:
            sen_list.append(str)
    return sen_list


# def find_words(input_list):
#     ln=len(input_list)
#     sentence_list=[]
#     for j in range(ln ):
#         end = min(j + 4, ln)
#         lst= [' '.join(input_list[j:i]) for i in range(j+1,end+1)]
#         for str in lst:
#             if str not in sentence_list:
#                 sentence_list.append(str)
#     return sentence_list
x_text, Y = load_data_and_y_labels("data/rt-polaritydata/ADE-positive-org.txt",
                                                    "data/rt-polaritydata/ADE-negative-org.txt")


f = open("ADR/ADR_string.txt", "r") #opens file with name of "test.txt"
ades=f.read()

# d="Rivaroxaban 2/2 lower back pain. Not very PC but am crippled by this drug. Taking more paracetamols. Must ring for 'phone consultation."
d="19.32 day 20 Rivaroxaban diary. Still residual aches and pains; only had 4 paracetamol today."
clean=clean_str(d)
# corrected=SpellChecker(clean)
# print("corrected {}".format(corrected))
# terms = twokenize.tokenizeRawTweetText(clean)

# pos_tags = nltk.pos_tag(terms, 'universal')

# print pos_tags
# word_tokens = nltk.word_tokenize(clean_str(d.lower()))

# min_len = min(32, len(pos_tags))
# for i in range(0,min_len ):
#     list=find_words(word_tokens)

sentence_ade = []
result=re.search(r'\b{0}\b'.format('tendon'),ades)
# print list
p = Pool(4)

for ade_arr in p.map(generate_ade_feature, x_text):
    sentence_ade.append(ade_arr)
# for t in x_text:
#     print clean_str(t)
#     arr=generate_ade_feature(t)
#     sentence_ade.append(arr)

data_set="ADE"
cupy.save("dump/" + data_set + "/ade-no-stopwords", sentence_ade)

# print sentence_ade.shape


# tkn_len=len(word_tokens)

#if full match then 100% ADR
#if part match then maybe and ADR Half score
#No of ADRs in sentence * array
