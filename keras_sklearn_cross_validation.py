#! /usr/bin/env python
from keras.callbacks import Callback
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import numpy
import data_helpers
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
import os,argparse,csv,time
from sklearn.metrics import  precision_score, recall_score, accuracy_score, f1_score
from itertools import starmap

from keras_models import create_model2, ltsm_model, create_model, model3, model4
from plot_utils import plot_fold_results, plot
from utils import get_time_diff
import matplotlib
# Force matplotlib to not use any Xwindows backend.

# matplotlib.use('Agg')

def get_file_path(name):
    return os.path.abspath(os.path.join(out_dir, name))



class Metrics(Callback):
    def __init__(self, split_id, x_val, y_val, batch_size,file_name,features_num,filters_num):
        self.x_val = x_val
        self.y_val = y_val
        self.split_id=split_id
        self.batch_size=batch_size
        self.accuracy = []
        self.loss=[]
        self.avg_precision_weighted=[]
        self.avg_precision_macro = []
        self.avg_precision_micro = []

        self.avg_recall_weighted = []
        self.avg_recall_macro = []
        self.avg_recall_micro = []

        self.avg_f1score_weighted = []
        self.avg_f1score_macro = []
        self.avg_f1score_micro = []

        self.pos_precision = []
        self.neg_precision = []
        self.pos_recall = []
        self.neg_recall = []
        self.pos_f1_score = []
        self.neg_f1_score = []
        self.file_name=file_name
        self.elapsed_time=[]
        self.elapsed_time_formatted = []
        split_id_formatted = ("{0}".format(split_id))
        features_num_formatted = ("{0}-features-without-{1}".format(features_num,removed_name))
        filters_num_formatted = ("{0}-filters".format(filters_num))
        batch_size_formatted = ("{0}-batch".format(batch_size))
        mod_name = ("{0}".format(model_name.__name__))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "results", data_set, mod_name,features_num_formatted,filters_num_formatted,batch_size_formatted, split_id_formatted))

        self.out_csv = os.path.abspath(os.path.join(self.out_dir, self.file_name))
        # create directory if not exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        result_csv = open(self.out_csv, "wb")
        csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        csv_writer.writerow(
            ["Epoch", "Accuracy", 'Loss', 'Avg Precision', 'Avg Recall', 'Avg F1-score', "Positive Precision",
             "Negative Precision", "Positive Recall","Negative Recall",
             "Positive f-score", "Negative f-score", "Elapsed Time"])

    def on_epoch_end(self, epoch, logs={}):
        predict = self.model.predict_classes(self.x_val, batch_size=self.batch_size)
        # Average precision
        weighted_prec=precision_score(self.y_val, predict, 'weighted') * 100
        self.avg_precision_weighted.append( weighted_prec)
        micro_prec=precision_score(self.y_val, predict, 'micro') * 100
        self.avg_precision_micro.append( micro_prec)
        macro_prec=precision_score(self.y_val, predict, 'macro') * 100
        self.avg_precision_macro.append( macro_prec)

        weighted_recall = recall_score(self.y_val, predict, 'weighted') * 100
        self.avg_recall_weighted.append(weighted_recall)
        micro_recall = recall_score(self.y_val, predict, 'micro') * 100
        self.avg_recall_micro.append(micro_recall)
        macro_recall = recall_score(self.y_val, predict, 'macro') * 100
        self.avg_recall_macro.append(macro_recall)

        weighted_fscore = f1_score(self.y_val, predict, 'weighted') * 100
        self.avg_f1score_weighted.append(weighted_fscore)
        micro_fscore = f1_score(self.y_val, predict, 'micro') * 100
        self.avg_f1score_micro.append(micro_fscore)
        macro_fscore = f1_score(self.y_val, predict, 'macro') * 100
        self.avg_f1score_macro.append(macro_fscore)

        positive_prec = precision_score(self.y_val, predict, 'binary') * 100
        neg_prec = precision_score(self.y_val, predict, average='binary', pos_label=0) * 100
        self.pos_precision.append(positive_prec)
        self.neg_precision.append(neg_prec)

        pos_recall = recall_score(self.y_val, predict, 'binary') * 100
        neg_recall = recall_score(self.y_val, predict, average='binary', pos_label=0) * 100
        self.pos_recall.append(pos_recall)
        self.neg_recall.append(neg_recall)

        pos_fscore = f1_score(self.y_val, predict, 'binary') * 100
        neg_fscore = f1_score(self.y_val, predict, average='binary', pos_label=0) * 100
        self.pos_f1_score.append(pos_fscore )
        self.neg_f1_score.append(neg_fscore)
        self.loss.append(logs.get('loss'))
        acc=accuracy_score(self.y_val, predict)*100
        self.accuracy.append(acc)

        done = time.time()

        elapsed_formated = get_time_diff(self.start_time,done)
        elapsed_time=int((done-self.start_time)/60) # in minutes
        print elapsed_formated
        print elapsed_time
        self.elapsed_time.append(elapsed_time)
        self.elapsed_time_formatted.append(elapsed_formated)
        result_csv = open(self.out_csv, "a+")
        csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csv_writer.writerow(
            [epoch, acc, logs.get('loss'), weighted_prec,weighted_recall, weighted_fscore, positive_prec, neg_prec, pos_recall, neg_recall,
             pos_fscore, neg_fscore, elapsed_time])
        return


    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    # def on_train_end(self, logs=None):

        # plot(self.accuracy, self.avg_precision_weighted, self.elapsed_time, "Accuracy-Loss", "Time", "Score", ['Accuracy', 'Loss'], os.path.join(self.out_dir,"Test"))

        # result_csv = open(self.out_csv, "a+")
        # csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        #
        # for x in range(0, epochs_num):#, self.loss[x]
        #     csv_writer.writerow([x, self.accuracy[x],self.loss[x], self.avg_precision_weighted[x], self.avg_precision_macro[x], self.avg_precision_micro[x],
        #      self.avg_recall_weighted[x], self.avg_recall_macro[x], self.avg_recall_micro[x], self.avg_f1score_weighted[x], self.avg_f1score_macro[x],
        #      self.avg_f1score_micro,self.pos_precision,self.neg_precision,self.pos_recall,self.neg_recall, self.pos_f1_score,self.neg_f1_score, self.elapsed_time[x]])
        #
        # return


parser = argparse.ArgumentParser()
parser.add_argument("data_set", type=str,
                    help="Data set name",default='TWDS2')
parser.add_argument("features_num", type=int,
                    help="Features number",default=12)
parser.add_argument("removed_feature", type=int,
                    help="Remove Feature number",default=12)
parser.add_argument("ade_exact", type=int,
                    help="Remove Feature name",default=1)
parser.add_argument("removed_name", type=str,
                    help="Remove Feature name",default="ADE")
parser.add_argument("batch_size", type=int,
                    help="Batch size",default=128)
parser.add_argument("epochs_num", type=int,
                    help="Number of epochs",default=20)
parser.add_argument("splits_num", type=int,
                    help="splits_num",default=10)
args = parser.parse_args()
data_set = args.data_set
features_num=args.features_num
batch_size=args.batch_size
epochs_num = args.epochs_num
splits_num = args.splits_num
removed_feature= args.removed_feature
removed_name= args.removed_name
ade_exact=args.ade_exact
filter_sizes = (3, 8)

num_filters = 32
dropout_prob = (0.5, 0.8)
hidden_dims = 250
model_name=create_model



if  os.path.isfile("dump/" + data_set + "/negScore.npy") and os.stat("dump/" + data_set + "/negScore.npy").st_size > 0:
    negScore = numpy.load("dump/" + data_set + "/negScore.npy")
    posScore = numpy.load("dump/" + data_set + "/posScore.npy")
    subjScore = numpy.load("dump/" + data_set + "/subjScore.npy")
    pposScore = numpy.load("dump/" + data_set + "/pposScore.npy")
    nnegScore = numpy.load("dump/" + data_set + "/nnegScore.npy")
    moreGoodScore = numpy.load("dump/" + data_set + "/moreGoodScore.npy")
    moreBadScore = numpy.load("dump/" + data_set + "/moreBadScore.npy")
    lessBadScore = numpy.load("dump/" + data_set + "/lessBadScore.npy")
    lessGoodScore = numpy.load("dump/" + data_set + "/lessGoodScore.npy")
    sentence_cluster = numpy.load("dump/" + data_set + "/sentence_cluster.npy")
    sentence_cluster2 = numpy.load("dump/" + data_set + "/sentence_cluster2.npy")
    wordLength = numpy.load("dump/" + data_set + "/wordLength.npy")
    wordOrder = numpy.load("dump/" + data_set + "/wordOrder.npy")


    Y = numpy.load("dump/" + data_set + "/Y.npy")

    if data_set=='STS' or data_set=='MR':
        anger_feature_list=numpy.load("dump/" + data_set + "/anger_feature_list.npy" )
        anticipation_feature_list=numpy.load("dump/" + data_set + "/anticipation_feature_list.npy")
        disgust_feature_list=numpy.load("dump/" + data_set + "/disgust_feature_list.npy")
        fear_feature_list=numpy.load("dump/" + data_set + "/fear_feature_list.npy")
        joy_feature_list=numpy.load("dump/" + data_set + "/joy_feature_list.npy")
        negative_feature_list=numpy.load("dump/" + data_set + "/negative_feature_list.npy")
        positive_feature_list=numpy.load("dump/" + data_set + "/positive_feature_list.npy")
        sadness_feature_list=numpy.load("dump/" + data_set + "/sadness_feature_list.npy")
        surprise_feature_list=numpy.load("dump/" + data_set + "/surprise_feature_list.npy")
        trust_feature_list=numpy.load("dump/" + data_set + "/trust_feature_list.npy")
        # sentimentScore = numpy.load("dump/" + data_set + "/sentiment-score.npy")
        # sentimentScore= numpy.expand_dims(sentimentScore, axis=2)
        valenceScore=numpy.load("dump/" + data_set +"/valence_feature_list.npy")
        arousalScore = numpy.load("dump/" + data_set + "/arousal_feature_list.npy")
        dominanceScore = numpy.load("dump/" + data_set + "/dominance_feature_list.npy")
        # print sentimentScore.shape
    else:
        adeExact = numpy.load("dump/" + data_set + "/ade-exact.npy")
        # adeNoStopWords = numpy.load("dump/" + data_set + "/ade-no-stopwords.npy")

    # x_text, Y = data_helpers.load_data_and_labels("./data/rt-polaritydata/sts-gold.pos",
    #                                                 "./data/rt-polaritydata/sts-gold.neg")

    print Y.shape
else:
    # file = open("testfile.txt", "wb")
    x_text, Y = data_helpers.load_data_and_y_labels("./data/MR/rt-polarity.pos",
                                                    "./data/MR/rt-polarity.neg")
    FeatureExtractionUtilities.loadItems()

    max_document_length = 32

    s, negScore, posScore, adeScore, subjScore, pposScore, nnegScore, moreGoodScore, moreBadScore, lessBadScore, lessGoodScore, sentence_cluster, sentence_cluster2, wordLength, wordOrder = FeatureExtractionUtilities.generateSemVec(
        x_text, max_document_length, embed_size=1)

    s = numpy.array(s)
    negScore = numpy.array(negScore)
    posScore = numpy.array(posScore)
    adeScore = numpy.array(adeScore)
    subjScore = numpy.array(subjScore)
    pposScore = numpy.array(pposScore)
    nnegScore = numpy.array(nnegScore)
    moreGoodScore = numpy.array(moreGoodScore)
    moreBadScore = numpy.array(moreBadScore)
    lessBadScore = numpy.array(lessBadScore)
    lessGoodScore = numpy.array(lessGoodScore)
    sentence_cluster = numpy.array(sentence_cluster)
    sentence_cluster2 = numpy.array(sentence_cluster2)
    wordLength = numpy.array(wordLength)
    wordOrder = numpy.array(wordOrder)

    numpy.save("dump/" + data_set + "/negScore", negScore)
    numpy.save("dump/" + data_set + "/posScore", posScore)
    # numpy.save("dump/" + data_set + "/adeScore", adeScore)
    numpy.save("dump/" + data_set + "/subjScore", subjScore)
    numpy.save("dump/" + data_set + "/pposScore", pposScore)
    numpy.save("dump/" + data_set + "/nnegScore", nnegScore)
    numpy.save("dump/" + data_set + "/moreGoodScore", moreGoodScore)
    numpy.save("dump/" + data_set + "/moreBadScore", moreBadScore)
    numpy.save("dump/" + data_set + "/lessBadScore", lessBadScore)
    numpy.save("dump/" + data_set + "/lessGoodScore", lessGoodScore)
    numpy.save("dump/" + data_set + "/sentence_cluster", sentence_cluster)
    numpy.save("dump/" + data_set + "/sentence_cluster2", sentence_cluster2)
    numpy.save("dump/" + data_set + "/wordLength", wordLength)
    numpy.save("dump/" + data_set + "/wordOrder", wordOrder)
    numpy.save("dump/" + data_set + "/Y", Y)
# X = numpy.load("testfile.npy")


if data_set=='STS' or data_set=='MR':
    #nnegScore,negative_feature_list,moreBadScore, negScore,posScore, pposScore,  moreGoodScore, positive_feature_list,
    features_lst = [negScore, posScore, subjScore, pposScore, nnegScore, anger_feature_list,anticipation_feature_list,#valenceScore,arousalScore,dominanceScore,
                    disgust_feature_list, fear_feature_list, joy_feature_list, negative_feature_list,positive_feature_list,
                    sadness_feature_list,surprise_feature_list,trust_feature_list,
                    moreGoodScore, lessBadScore, lessGoodScore, moreBadScore, wordLength, wordOrder]

else:
    if ade_exact==1:
        adeExact = numpy.expand_dims(adeExact, axis=2)
        # adeNoStopWords = numpy.expand_dims(adeNoStopWords, axis=2)
        # features_lst = [negScore, posScore, subjScore, pposScore, nnegScore, #adeExact,
        #                     moreGoodScore, lessBadScore, lessGoodScore, moreBadScore, wordLength, wordOrder]
        features_lst=[posScore,negScore,adeExact,subjScore,pposScore,wordLength,moreBadScore,moreGoodScore, lessBadScore, lessGoodScore]#adeExact
    else:
        adeNoStopWords = numpy.expand_dims(adeNoStopWords, axis=2)
        features_lst = [negScore, posScore, subjScore, pposScore, nnegScore,
                        moreGoodScore, lessBadScore, lessGoodScore, moreBadScore, wordLength, wordOrder]

    # ade2 = numpy.expand_dims(ade2, axis=2)
    # print negScore.shape
    # print ade2.shape

# if removed_feature <13:
#     del features_lst[removed_feature]
#
#
X = numpy.concatenate(features_lst, axis=2)
if removed_feature!=13:
    X = numpy.expand_dims(X, axis=3)
    clust = numpy.expand_dims(sentence_cluster, axis=3)
    clust2 = numpy.expand_dims(sentence_cluster2, axis=3)
    X = numpy.concatenate([X, clust, clust2], axis=2)
    X=numpy.squeeze(X, axis=3)

print X.shape

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

kfold = StratifiedKFold(n_splits=splits_num, shuffle=True, random_state=seed)
model_accuracy_scores = []
metrics_acc_scores=[]
metrics_pos_precision_scores=[]
metrics_neg_precision_scores=[]

metrics_pos_recall_scores=[]
metrics_neg_recall_scores=[]

metrics_pos_fscore_scores=[]
metrics_neg_fscore_scores=[]

fold_micro_fscore_scores = []
fold_macro_fscore_scores = []
fold_weighted_fscore_scores=[]

fold_micro_precision_scores = []
fold_macro_precision_scores = []
fold_weighted_prec_scores=[]
fold_micro_recall_scores = []
fold_macro_recall_scores = []
fold_weighted_recall_scores=[]
fold_neg_prec_scores=[]
fold_positive_prec_scores=[]
fold_neg_recall_scores=[]
fold_positive_recall_scores=[]

fold_neg_fscore_scores=[]
fold_positive_fscore_scores=[]
epochs_arr = numpy.array(numpy.arange(1, epochs_num + 1))

# prec_y_data_list=[]

lw_lst=[]
alpha_lst=[]

fold_pos_prec_mean=[]
fold_neg_prec_mean=[]
fold_pos_recall_mean=[]
fold_neg_recall_mean=[]

fold_pos_fscore_mean=[]
fold_neg_fscore_mean=[]

fold_accuracy_mean=[]

index=0
for train, test in kfold.split(X, Y):
    print('index= %s'%index)
    test_y = np_utils.to_categorical(Y[test])
    train_y=np_utils.to_categorical(Y[train])
    # create model
    model = create_model(features_num)#create_model(features_num)
    file_name="%s-features-%s-filters.csv" % (features_num, num_filters)

    metrics=Metrics( index, X[test], Y[test], batch_size,file_name,features_num, num_filters)
    history=model.fit(X[train], Y[train], epochs=epochs_num, batch_size=batch_size, verbose=1, callbacks=[metrics])#, callbacks=[metrics], callbacks=[metrics,csv_logger]

    metrics_pos_precision_scores.append(metrics.pos_precision)
    metrics_neg_precision_scores.append(metrics.neg_precision)

    metrics_pos_fscore_scores.append(metrics.pos_f1_score)
    metrics_neg_fscore_scores.append(metrics.neg_f1_score)

    pos_prec_arr=numpy.array(metrics.pos_precision)
    precision_positive_mean=numpy.mean(metrics.pos_precision)
    fold_pos_prec_mean.append(precision_positive_mean)

    precision_neg_mean = numpy.mean(metrics.neg_precision)
    fold_neg_prec_mean.append(precision_neg_mean)

    recall_positive_mean = numpy.mean(metrics.pos_recall)
    fold_pos_recall_mean.append(recall_positive_mean)

    recall_neg_mean = numpy.mean(metrics.neg_recall)
    fold_neg_recall_mean.append(recall_neg_mean)

    fscore_positive_mean = numpy.mean(metrics.pos_f1_score)
    fold_pos_fscore_mean.append(fscore_positive_mean)

    fscore_neg_mean = numpy.mean(metrics.neg_f1_score)
    fold_neg_fscore_mean.append(fscore_neg_mean)


    metrics_neg_recall_scores.append(metrics.neg_recall)
    metrics_pos_recall_scores.append(metrics.pos_recall)



    # prec_y_data_list.append(pos_prec_arr)
    alpha_lst.append(0.3)
    lw_lst.append(1)


    # evaluate the model
    scores = model.evaluate(X[test],  Y[test], verbose=1)

    ypreds = model.predict_classes(X[test], batch_size=batch_size, verbose=1)

    positive_prec = precision_score(Y[test], ypreds, 'binary') * 100
    neg_prec = precision_score(Y[test], ypreds, average='binary',pos_label=0) * 100
    micro_prec = precision_score(Y[test], ypreds,'micro') * 100
    macro_prec = precision_score(Y[test], ypreds, 'macro') * 100
    weighted_prec = precision_score(Y[test], ypreds, 'weighted') * 100
    #
    pos_recall = recall_score(Y[test], ypreds, 'binary') * 100
    neg_recall = recall_score(Y[test], ypreds, average='binary', pos_label=0) * 100
    micro_recall = recall_score(Y[test], ypreds, 'micro') * 100
    macro_recall = recall_score(Y[test], ypreds, 'macro') * 100
    weighted_recall = recall_score(Y[test], ypreds, 'weighted') * 100
    fold_positive_prec_scores.append(positive_prec)
    fold_micro_precision_scores.append(micro_prec)
    fold_macro_precision_scores.append(macro_prec)
    fold_weighted_prec_scores.append(weighted_prec)
    fold_neg_prec_scores.append(neg_prec)

    pos_fscore = f1_score(Y[test], ypreds, 'binary') * 100
    neg_fscore = f1_score(Y[test], ypreds, average='binary', pos_label=0) * 100
    micro_fscore = f1_score(Y[test], ypreds, 'micro') * 100
    macro_fscore = f1_score(Y[test], ypreds, 'macro') * 100
    weighted_fscore = f1_score(Y[test], ypreds, 'weighted') * 100

    fold_micro_fscore_scores.append(micro_fscore)
    fold_macro_fscore_scores.append(macro_fscore)
    fold_weighted_fscore_scores.append(weighted_fscore)
    fold_neg_fscore_scores.append(neg_fscore)
    fold_positive_fscore_scores.append(pos_fscore)

    #
    fold_neg_recall_scores.append(neg_recall)
    fold_positive_recall_scores.append(pos_recall)
    fold_micro_recall_scores.append(micro_recall)
    fold_macro_recall_scores.append(macro_recall)
    fold_weighted_recall_scores.append(weighted_recall)

    model_accuracy_scores.append(scores[1] * 100)
    metrics_acc_scores.append(metrics.accuracy)
    fol_acc_mean=numpy.mean(metrics.accuracy)
    fold_accuracy_mean.append(fol_acc_mean)
    index=index+1


features_num_formatted = ("{0}-features-without-{1}".format(features_num,removed_name))
filters_num_formatted = ("{0}-filters".format(num_filters))
batch_size_formatted = ("{0}-batch".format(batch_size))
mod_name = ("{0}".format(model_name.__name__))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "results", data_set, mod_name,features_num_formatted,filters_num_formatted,batch_size_formatted))

out_prec_file = os.path.abspath(os.path.join(out_dir, 'positive_precision.png'))
neg_prec_file = os.path.abspath(os.path.join(out_dir, 'negative_precision.png'))

out_prec_model_file = os.path.abspath(os.path.join(out_dir, 'pos_precision_model_score.png'))
neg_prec_model_file = os.path.abspath(os.path.join(out_dir, 'neg_precision_model_score.png'))


out_accuracy_model_file = os.path.abspath(os.path.join(out_dir, 'model_accuracy.png'))

out_recall_file = os.path.abspath(os.path.join(out_dir, 'positive_recall.png'))
neg_recall_file = os.path.abspath(os.path.join(out_dir, 'negative_recall.png'))

out_fscore_file = os.path.abspath(os.path.join(out_dir, 'positive_fscore.png'))
neg_fscore_file = os.path.abspath(os.path.join(out_dir, 'negative_fscore.png'))

out_acc_file = os.path.abspath(os.path.join(out_dir, 'accuracy.png'))
pos_precision_acc = numpy.mean(metrics_pos_precision_scores, axis=0)
# prec_y_data_list.append(pos_precision_acc)


# prec_y_data_list
plot(epochs_arr, metrics_pos_precision_scores, 'Positive Precision', 'Epoch', 'Precision', epochs_num, out_prec_file, lw_lst, alpha_lst, fold_pos_prec_mean)

plot(epochs_arr, metrics_neg_precision_scores, 'Negative Precision', 'Epoch', 'Precision', epochs_num, neg_prec_file, lw_lst, alpha_lst, fold_neg_prec_mean)

plot(epochs_arr, metrics_acc_scores, 'Accuracy', 'Epoch', 'Accuracy', epochs_num, out_acc_file, lw_lst, alpha_lst, fold_accuracy_mean, [70, 101])

plot(epochs_arr, metrics_pos_recall_scores, 'Positive Recall', 'Epoch', 'Recall', epochs_num, out_recall_file, lw_lst, alpha_lst, fold_pos_recall_mean)

plot(epochs_arr, metrics_neg_recall_scores, 'Negative Recall', 'Epoch', 'Recall', epochs_num, neg_recall_file, lw_lst, alpha_lst, fold_neg_recall_mean)

plot(epochs_arr, metrics_pos_fscore_scores, 'Positive F-score', 'Epoch', 'F-score', epochs_num, out_fscore_file, lw_lst, alpha_lst, fold_pos_fscore_mean)

plot(epochs_arr, metrics_neg_fscore_scores, 'Negative F-score', 'Epoch', 'F-score', epochs_num, neg_fscore_file, lw_lst, alpha_lst, fold_neg_fscore_mean)



# Micro vs Macro
# If you think all the labels are more or less equally sized (have roughly the same number of instances), use any.

# If you think there are labels with more instances than others and if you want to bias your metric towards the most populated ones, use micromedia.

# If you think there are labels with more instances than others and if you want to bias your metric toward the least populated ones (or at least you don't want to bias toward the most populated ones), use macromedia.

# If the micromedia result is significantly lower than the macromedia one, it means that you have some gross misclassification in the most populated labels, whereas your smaller labels are probably correctly classified. If the macromedia result is significantly lower than the micromedia one, it means your smaller labels are poorly classified, whereas your larger ones are probably correctly classified.

# http://www.sciencedirect.com/science/article/pii/S0306457309000259

# A systematic analysis of performance measures for classification tasks

#TODO: CSV file with final results

plot_fold_results(model_accuracy_scores, 'Model Accuracy', 'Accuracy', out_accuracy_model_file)

plot_fold_results(fold_positive_prec_scores, 'Model Positive Precision', 'Precision', out_prec_model_file)
plot_fold_results(fold_neg_prec_scores, 'Model Negative Precision', 'Precision', neg_prec_model_file)

plot_fold_results(fold_neg_fscore_scores, 'Model Negative F-score', 'F-score',get_file_path('model_neg_fscore.png') )

plot_fold_results(fold_positive_fscore_scores, 'Model Positive F-score', 'F-score',get_file_path('model_pos_fscore.png') )

plot_fold_results(fold_neg_recall_scores, 'Model Negative Recall', 'Recall',get_file_path('model_neg_recall.png') )

plot_fold_results(fold_positive_recall_scores, 'Model Positive Recall', 'Recall',get_file_path('model_pos_recall.png') )


plot_fold_results(fold_macro_fscore_scores, 'Model Macro F-score', 'F-score', get_file_path('model_macro_fscore.png'))

plot_fold_results(fold_micro_fscore_scores, 'Model Micro F-score', 'F-score', get_file_path('model_micro_fscore.png'))

plot_fold_results(fold_weighted_fscore_scores, 'Model Weighted F-score', 'F-score', get_file_path('model_weighted_fscore.png'))

plot_fold_results(fold_micro_precision_scores, 'Model Micro Precision', 'Precision', get_file_path('model_micro_precision.png'))
plot_fold_results(fold_macro_precision_scores, 'Model Macro Precision', 'Precision', get_file_path('model_macro_precision.png'))
plot_fold_results(fold_weighted_prec_scores, 'Model Weighted Precision', 'Precision', get_file_path('model_weighted_precision.png'))

plot_fold_results(fold_micro_recall_scores, 'Model Micro Recall', 'Precision', get_file_path('model_micro_recall.png'))
plot_fold_results(fold_macro_recall_scores, 'Model Macro Recall', 'Precision', get_file_path('model_macro_recall.png'))
plot_fold_results(fold_weighted_recall_scores, 'Model Weighted Recall', 'Precision', get_file_path('model_weighted_recall.png'))



model_out_csv = os.path.abspath(os.path.join(out_dir, 'model_results.csv'))
        # create directory if not exist
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

result_csv = open(model_out_csv, "wb")
csv_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

csv_writer.writerow(["Fold", "Accuracy", 'Avg Precision Weighted', 'Avg Precision Macro', 'Avg Precision Micro',
             'Avg Recall Weighted', 'Avg Recall Macro', 'Avg Recall Micro', 'Avg F1-score Weighted',
             'Avg F1-score Macro', 'Avg F1-score Micro', "Positive Precision", "Negative Precision", "Positive Recall",
             "Negative Recall","Positive f-score", "Negative f-score"])

for x in range(0, splits_num):  # , self.loss[x]
    csv_writer.writerow([x, model_accuracy_scores[x], fold_weighted_prec_scores[x], fold_macro_precision_scores[x],
                         fold_micro_precision_scores[x], fold_weighted_recall_scores[x], fold_macro_recall_scores[x], fold_micro_recall_scores[x],
                         fold_weighted_fscore_scores[x], fold_macro_fscore_scores[x],
                         fold_micro_fscore_scores[x], fold_positive_prec_scores[x], fold_neg_prec_scores[x]
                         , fold_positive_recall_scores[x], fold_neg_recall_scores[x], fold_positive_fscore_scores[x], fold_neg_fscore_scores[x]
                         # self.neg_recall, self.pos_f1_score, self.neg_f1_score, self.elapsed_time[x]
                         ])

print("\n%.2f%% (+/- %.2f%%)" % (numpy.mean(model_accuracy_scores), numpy.std(model_accuracy_scores)))
print("===Positive precision===")
print("Positive precision %s" % ' '.join(starmap('{}:{}'.format, enumerate(fold_positive_prec_scores))))
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(fold_positive_prec_scores), numpy.std(fold_positive_prec_scores)))


print("===Negative precision===")
print("Negative precision %s" % ' '.join(starmap('{}:{}'.format, enumerate(fold_neg_prec_scores))))
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(fold_neg_prec_scores), numpy.std(fold_neg_prec_scores)))

