import csv

neg_file = open("data/sts-gold.neg", "w")
pos_file= open("data/sts-gold.pos", "w")

with open('data/sts_gold_tweet.csv', 'rb') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
  for row in reader:
      if row['"polarity"']== '"0"':
          neg_file.write(row['"tweet"'][1:-1]+'\n')
      if row['"polarity"'] == '"4"':
          pos_file.write(row['"tweet"'][1:-1]+'\n')

pos_file.close()
neg_file.close()