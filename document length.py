import nltk

file = open("data/rt-polaritydata/ADE-negative-org.txt", "r")
for line in file:
    max=len(nltk.pos_tag(line, 'universal'))
    print max