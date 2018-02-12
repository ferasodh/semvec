import json,re,os,csv

root_dir = 'results/TWDS5/create_model'
res_path='results/TWDS2/grid-search-results_create_model.txt'

result_csv = open(os.path.join(root_dir,'grid_search.csv'), "wb")
csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

csv_writer.writerow(
    ["Accuracy","Loss","Epochs","Optimizer","Hidden Dimensions","Batch Size"])

with open(res_path) as f:
    for line in f.readlines():
        acc = round(float(line[0:line.index(" (")])*100,2)

        loss = line[line.index("(") + 1:line.index(")")]
        print loss
        start = line.index("{")

        js = line[start:]

        jsn = json.loads(js)
        print jsn["epochs"]

        csv_writer.writerow([acc,loss,jsn["epochs"],jsn["optimizer"],jsn["hidden_dims"],jsn["batch_size"]])

