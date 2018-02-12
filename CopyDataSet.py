import os
path = '/Volumes/MACHD/Feras/Downloads/aclImdb/train/pos'
files = os.listdir(path)

with open('/Volumes/MACHD/Feras/Documents/Master/Thesis/Experiment/ADRBinaryClassifier/data/rt-polaritydata/imdb.pos', 'w') as outfile:
    for fname in files:
        full_file_name = os.path.join(path, fname)
        with open(full_file_name) as infile:
            outfile.write(infile.read()+ '\n')