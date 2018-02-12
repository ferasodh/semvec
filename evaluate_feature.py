import string

from data_helpers import clean_str
from featureextractionmodules.FeatureExtractionUtilities import stemmer

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/MR/rt-polarity.neg").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/MR/rt-polarity.pos").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    # x_text = positive_examples + negative_examples
    positive_examples = [clean_str(sent) for sent in positive_examples]
    positive_examples = [s.split(" ") for s in positive_examples]

    negative_examples = [clean_str(sent) for sent in negative_examples]
    negative_examples = [s.split(" ") for s in negative_examples]

    # Generate labels

    return [positive_examples, negative_examples]



posWords = []
infile = open('lexicons/PosNegWords/pos_mod.txt')
for line in infile:
    if not line[0] == ';':
        posWords.append(stemmer.stem(string.strip(line.decode('utf8', 'ignore').encode('ascii', 'ignore'))))

negWords=[]
infile = open('lexicons/PosNegWords/neg_mod.txt')
for line in infile:
    if not line[0] == ';':
        negWords.append(stemmer.stem(string.strip(line.decode('utf8', 'ignore').encode('ascii', 'ignore'))))


pos, neg=load_data_and_labels()

c=0
n=0
for sent in neg:

    for w in sent:
        if w in posWords:
            c=c+1
        if w in negWords:
            n=n+1

print c
print n



