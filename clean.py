import re

devs = list(open('data/SemEval2015-dev.tsv', "r").readlines())
cleanFile = open("data/SemEval2015-dev-clean.tsv", 'w')


for ex in devs:
    # if ex=='Not Available':
    #     continue
    strs = re.split(r'\t+', ex)
    if strs[3].strip() != 'Not Available':
        cleanFile.write(ex)


devs = list(open('data/dev.csv', "r").readlines())
cleanFile = list(open("data/SemEval2015-dev-clean.tsv", 'r').readlines())

cleaned=[]
for line in cleanFile:
    strs = re.split(r'\t+', line)
    cleaned.append(strs[0])
# cleanFile.close()

cleanFile = open("data/SemEval2015-dev-clean.tsv", 'ab')

for ex in devs:
    # if ex=='Not Available':
    #     continue
    strs = re.split(r'\t+', ex)
    if strs[0].strip() not in cleaned:
        cleanFile.write("%s\t%s\t%s\t%s\t\n" % (strs[0],'1', strs[3].strip(), strs[1].strip() ))

cleanFile.close()
devs.close()