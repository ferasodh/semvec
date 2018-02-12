#!/usr/local/bin/python
# coding: UTF-8
import csv,os
import numpy as np

def generate_results_csv(result_csv,file_obj,label):
    """

    Read a CSV file using csv.DictReader

    """

    reader = csv.DictReader(file_obj, delimiter=',')
    accuracy_arr= np.zeros(10)
    pos_fscore_arr=np.zeros(10)
    neg_fscore_arr = np.zeros(10)
    pos_precision_arr = np.zeros(10)
    pos_recall_arr = np.zeros(10)

    i=0
    for line in reader:
        accuracy_arr[i]=line["Accuracy"]
        pos_fscore_arr[i] = line["Positive f-score"]
        neg_fscore_arr[i] = line["Negative f-score"]
        pos_precision_arr[i]=line["Positive Precision"]
        pos_recall_arr[i] = line["Positive Recall"]
        i=i+1

    mean_acc= round(np.mean(accuracy_arr),2)
    std_acc= round(np.std(accuracy_arr),2)
    mean_pos_fscore= round(np.mean(pos_fscore_arr), 2)
    std_pos_fscore= round(np.std(pos_fscore_arr), 2)
    mean_neg_fscore= round(np.mean(neg_fscore_arr),2)
    std_neg_fscore= round(np.std(neg_fscore_arr),2)
    mean_pos_precision= round(np.mean(pos_precision_arr), 2)
    std_pos_precision= round(np.std(pos_precision_arr), 2)
    mean_pos_recall= round(np.mean(pos_recall_arr), 2)
    std_pos_recall= round(np.std(pos_recall_arr), 2)
    csv_writer.writerow([label,mean_acc,std_acc,mean_pos_fscore,std_pos_fscore,mean_neg_fscore,std_neg_fscore,mean_pos_precision,std_pos_precision,mean_pos_recall,std_pos_recall])


if __name__ == "__main__":

    dirs=['32-features-ade-exact','31-features-without-wordOrder','31-features-without-wordLength','31-features-without-subjScore',
          '31-features-without-pposScore','31-features-without-posScore','31-features-without-nnegScore','31-features-without-negScore',
          '31-features-without-moreGoodScore','31-features-without-moreBadScore','31-features-without-lessGoodScore','31-features-without-lessBadScore',
          '31-features-without-adeExact','12-features-without-word-cluster']
    labels=['All','Word Order','Word Length','Subjectivity','Positive SentiWordNet','Positive Opinion','Negative SentiWordNet','Negative Opinion',
            'More Good','More Bad','Less Good','Less Bad','ADR','Word Cluster']

    root_dir = 'results/TWDS5/create_model'
    last_path='32-filters/128-batch/model_results.csv'

    result_csv = open(os.path.join(root_dir,'features_importance.csv'), "wb")
    csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    csv_writer.writerow(
        ["Features","Mean (%)","SD (±)","Mean (%)","SD (±)","Mean (%)","SD (±)","Mean (%)","SD (±)","Mean (%)","SD (±)"])

    for i in range(0,13):
        file_path=os.path.join(root_dir,dirs[i],last_path)
        print file_path
        with open(file_path) as f_obj:
            generate_results_csv(result_csv,f_obj,labels[i])

