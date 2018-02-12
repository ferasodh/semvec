




goldFile = open("./data/SemEval2015-task10-test-B-input.txt", 'r')
predFile = open("semeval2015_results_out.txt", 'w')
resFile = open("semeval2015_results.txt", 'r')

predictLabel=[]
for line in resFile:
    if line =='0\n':
        predictLabel.append('neutral')
    if line =='1\n':
        predictLabel.append('positive')
    if line =='2\n':
        predictLabel.append('negative')



index = 0
for line in goldFile:
    data = line.strip("\r\n").split("\t")
    predFile.write("%s\t%s\t%s\n" % (data[0], data[1], predictLabel[index]))
    index += 1

goldFile.close()
predFile.close()


