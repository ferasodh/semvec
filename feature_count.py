from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from EmoticonDetector import EmoticonDetector

positive_examples = list(open("./data/sts-gold.pos").readlines())
positive_examples = [s.strip() for s in positive_examples]
negative_examples = list(open("./data/sts-gold.neg").readlines())
negative_examples = [s.strip() for s in negative_examples]
# Split by words
# x_text = positive_examples + negative_examples
# x_text = [clean_str(sent) for sent in x_text]
# x_text = [s.split(" ") for s inin x_text]
pos_count=0
analyzer = SentimentIntensityAnalyzer()
for line in negative_examples:
    # for s in line.split(" "):
        vs = analyzer.polarity_scores(line)
        print "{:-<65} {}".format( str(vs),line)

print pos_count