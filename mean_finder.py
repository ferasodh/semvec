import csv
from collections import OrderedDict, defaultdict, Counter
import numpy,os

valenceList = defaultdict(list)
arousalList = defaultdict(list)
dominanceList=defaultdict(list)
fscore_list=[]
precision_list=[]
recall_list=[]
acc_list=[]
for i in range(9):
  fname=  'results/TWDS5/create_model/31-features-without-adeExact/32-filters/128-batch/%s/31-features-32-filters.csv'% (i)
  with open(fname, 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    idx=0
    accs=[]
    for line in reader:
        if idx==0:
            str=line["Positive f-score"]
            str=str.replace("[", "")
            str=str.replace("]", "")

            fscore = numpy.fromstring(str, dtype=numpy.float, sep=',')
            fscore_list.append(fscore)

            str = line["Positive Recall"]
            str = str.replace("[", "")
            str = str.replace("]", "")

            recall = numpy.fromstring(str, dtype=numpy.float, sep=',')
            recall_list.append(recall)

            str = line["Positive Precision"]
            str = str.replace("[", "")
            str = str.replace("]", "")

            prec = numpy.fromstring(str, dtype=numpy.float, sep=',')
            precision_list.append(prec)

        accs.append(float(line["Accuracy"]))
        idx=idx+1
    accs=numpy.asarray(accs)
    acc_list.append(accs)


print 'avg'

mean_fscore=numpy.mean( numpy.array(fscore_list), axis=0 )
mean_recall=numpy.mean( numpy.array(recall_list), axis=0 )
mean_prec=numpy.mean( numpy.array(precision_list), axis=0 )
mean_acc=numpy.mean( numpy.array(acc_list), axis=0 )
out_csv = os.path.abspath(os.path.join('results/TWDS5/create_model/31-features-without-adeExact/32-filters/128-batch', 'train_mean.csv'))

result_csv = open(out_csv, "wb")
csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

csv_writer.writerow(
            ["Epoch", "Accuracy", 'Avg Precision', 'Avg Recall', 'Avg F1-score'])

for x in range(0, 300):#, self.loss[x]
            csv_writer.writerow([x, mean_acc[x],mean_prec[x], mean_recall[x], mean_fscore[x]])