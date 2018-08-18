#! /usr/bin/env python
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV
import cupy
from keras.layers import  Dense, Dropout, Activation, Flatten,Conv1D, MaxPooling1D, LSTM
import data_helpers
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
import os, argparse, sys



def create_model2(features_num=12,hidden_dims=128,optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    # model.add(Embedding(20000, 128, input_length=300))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, activation='relu',input_shape=(60,features_num),kernel_initializer=init))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_dims,kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid',kernel_initializer=init))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print("model fitting - simplified convolutional neural network")
    # model.summary()
    return model

# Function to create model, required for KerasClassifier
def create_model(features_num=12,hidden_dims=128,optimizer='rmsprop',filters=64,kernel_size=2,pool_size=2,dropout=0.7):
    input_shape = (32, features_num)
    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=input_shape, filters=filters, kernel_size=kernel_size))

    # we use max pooling:
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    # model.add(LSTM(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print model.summary()
    return model


parser = argparse.ArgumentParser()
parser.add_argument("data_set", type=str,
                    help="Data set name",default='STS')
parser.add_argument("task_number", type=int,
                    help="Task number",default=1)


args = parser.parse_args()
data_set = args.data_set
task_number=args.task_number

# sys.stdout = open("results/" + data_set+'/grid_search_results.txt', 'a+')


if os.path.isfile("dump/" + data_set + "/negScore.npy") and os.stat("dump/" + data_set + "/negScore.npy").st_size > 0:
    negScore = cupy.load("dump/" + data_set + "/negScore.npy")
    posScore = cupy.load("dump/" + data_set + "/posScore.npy")
    adeScore = cupy.load("dump/" + data_set + "/adeScore.npy")
    subjScore = cupy.load("dump/" + data_set + "/subjScore.npy")
    pposScore = cupy.load("dump/" + data_set + "/pposScore.npy")
    nnegScore = cupy.load("dump/" + data_set + "/nnegScore.npy")
    moreGoodScore = cupy.load("dump/" + data_set + "/moreGoodScore.npy")
    moreBadScore = cupy.load("dump/" + data_set + "/moreBadScore.npy")
    lessBadScore = cupy.load("dump/" + data_set + "/lessBadScore.npy")
    lessGoodScore = cupy.load("dump/" + data_set + "/lessGoodScore.npy")
    sentence_cluster = cupy.load("dump/" + data_set + "/sentence_cluster.npy")
    sentence_cluster2 = cupy.load("dump/" + data_set + "/sentence_cluster2.npy")
    wordLength = cupy.load("dump/" + data_set + "/wordLength.npy")
    wordOrder = cupy.load("dump/" + data_set + "/wordOrder.npy")
    Y = cupy.load("dump/" + data_set + "/Y.npy")
else:
    # file = open("testfile.txt", "wb")
    x_text, Y = data_helpers.load_data_and_y_labels("./data/rt-polaritydata/sts-gold.pos",
                                                    "./data/rt-polaritydata/sts-gold.neg")
    FeatureExtractionUtilities.loadItems()

    max_document_length = 60

    s, negScore, posScore, adeScore, subjScore, pposScore, nnegScore, moreGoodScore, moreBadScore, lessBadScore, lessGoodScore, sentence_cluster, sentence_cluster2, wordLength, wordOrder = FeatureExtractionUtilities.generateSemVec(
        x_text, max_document_length, embed_size=1)

    s = cupy.array(s)
    negScore = cupy.array(negScore)
    posScore = cupy.array(posScore)
    adeScore = cupy.array(adeScore)
    subjScore = cupy.array(subjScore)
    pposScore = cupy.array(pposScore)
    nnegScore = cupy.array(nnegScore)
    moreGoodScore = cupy.array(moreGoodScore)
    moreBadScore = cupy.array(moreBadScore)
    lessBadScore = cupy.array(lessBadScore)
    lessGoodScore = cupy.array(lessGoodScore)
    sentence_cluster = cupy.array(sentence_cluster)
    sentence_cluster2 = cupy.array(sentence_cluster2)
    wordLength = cupy.array(wordLength)
    wordOrder = cupy.array(wordOrder)

    cupy.save("dump/" + data_set + "/negScore", negScore)
    cupy.save("dump/" + data_set + "/posScore", posScore)
    cupy.save("dump/" + data_set + "/adeScore", adeScore)
    cupy.save("dump/" + data_set + "/subjScore", subjScore)
    cupy.save("dump/" + data_set + "/pposScore", pposScore)
    cupy.save("dump/" + data_set + "/nnegScore", nnegScore)
    cupy.save("dump/" + data_set + "/moreGoodScore", moreGoodScore)
    cupy.save("dump/" + data_set + "/moreBadScore", moreBadScore)
    cupy.save("dump/" + data_set + "/lessBadScore", lessBadScore)
    cupy.save("dump/" + data_set + "/lessGoodScore", lessGoodScore)
    cupy.save("dump/" + data_set + "/sentence_cluster", sentence_cluster)
    cupy.save("dump/" + data_set + "/sentence_cluster2", sentence_cluster2)
    cupy.save("dump/" + data_set + "/wordLength", wordLength)
    cupy.save("dump/" + data_set + "/wordOrder", wordOrder)
    cupy.save("dump/" + data_set + "/Y", Y)
# X = cupy.load("testfile.npy")

X = cupy.concatenate([negScore, posScore,  subjScore, pposScore, nnegScore, adeScore,
                       moreGoodScore, moreBadScore, lessBadScore, lessGoodScore, wordLength, wordOrder], axis=2)

    # cupy.save("testfile", X)
    # cupy.save("testfiley",Y)


# X=cupy.expand_dims(X, axis=3)



# fix random seed for reproducibility
seed = 7
cupy.random.seed(seed)
# load pima indians dataset

pool_size=[2]
kernel_size=[2]
filters=[64]
dropout=[0.7]

# create model
model = KerasClassifier(build_fn=create_model, verbose=1)
# grid search epochs, batch size and optimizer
if data_set=='STS':
    if task_number==1:
        optimizers = ['RMSprop']
        # ltsm_units = [64]
        hidden_dims= [64]
        epochs = [20,50,100]
        batches = [20, 50, 100]#
    if task_number==2:
        optimizers = ['RMSprop']
        # ltsm_units = [128]
        hidden_dims = [128]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number==3:
        optimizers = ['RMSprop']
        # ltsm_units = [256]
        hidden_dims = [256]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number==4:
        optimizers = ['Adam']
        # ltsm_units = [64]
        hidden_dims = [256]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number==5:
        optimizers = ['Adam']
        # ltsm_units = [128]
        hidden_dims = [128]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number==6:
        optimizers = ['Adam']
        # ltsm_units = [256]
        hidden_dims = [256]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number == 7:
        optimizers = ['RMSprop']
        # ltsm_units = [32]
        hidden_dims = [32]
        epochs = [20,50,100]
        batches = [20, 50, 100]
    if task_number == 8:
        optimizers = ['Adam']
        # ltsm_units = [32]
        hidden_dims = [32]
        epochs = [20,50,100]
        batches = [20, 50, 100]

else:
    if task_number==1:
        optimizers = ['RMSprop']
        hidden_dims = [64]
        epochs = [100,200,300]
        batches = [64, 128,256]#128,

    if task_number == 2:
        optimizers = ['RMSprop']
        hidden_dims = [128] #128
        epochs = [100,200,300]
        batches = [ 64, 128, 256]

    if task_number == 3:
        optimizers = ['RMSprop']
        hidden_dims = [256] #128
        epochs = [100, 200, 300]
        batches = [ 64, 128, 256]

    if task_number == 4:
        optimizers = ['Adam']
        hidden_dims = [64]
        epochs = [100,200,300]
        batches = [64, 128,256]#128,

    if task_number == 5:
        optimizers = ['Adam']
        hidden_dims = [128]
        epochs = [100,200,300]
        batches = [64, 128,256]#128,

    if task_number == 6:
        optimizers = ['Adam']
        hidden_dims = [256]
        epochs = [100,200,300]
        batches = [64, 128,256]#128,

    if task_number == 7:
        optimizers = ['Adam']
        hidden_dims = [128]
        epochs = [300]
        batches = [128]
        filters=[32]
        dropout=[0.5,0.7,0.8]
        kernel_size=[2,3,4]
        pool_size=[2,3,4]
    if task_number == 8:
        optimizers = ['Adam']
        hidden_dims = [128]
        epochs = [300]
        batches = [128]
        filters = [ 64]
        dropout = [0.5, 0.7, 0.8]
        kernel_size = [2, 3, 4]
        pool_size = [2, 3, 4]
    if task_number == 9:
        optimizers = ['Adam']
        hidden_dims = [128]
        epochs = [300]
        batches = [128]
        filters = [128]
        dropout = [0.5, 0.7, 0.8]
        kernel_size = [2, 3, 4]
        pool_size = [2, 3, 4]



param_grid = dict( epochs=epochs, batch_size=batches, optimizer=optimizers, hidden_dims=hidden_dims,pool_size=pool_size,kernel_size=kernel_size,filters=filters,dropout=dropout)# , init=init
grid = GridSearchCV(cv=10,estimator=model,n_jobs=4, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))