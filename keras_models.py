from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv1D, MaxPooling1D, Merge, LSTM, Convolution1D, \
    GlobalAveragePooling1D, Reshape, Convolution2D, MaxPooling2D, merge, Embedding, Concatenate
from keras.optimizers import Adam, SGD


def create_model2(features_num):
    model = Sequential()
    # model.add(Embedding(20000, 128, input_length=300))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, activation='relu',input_shape=(32,features_num)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("model fitting - simplified convolutional neural network")
    # model.summary()
    return model

def ltsm_model(features_num,units=64,optimizer='RMSprop'):
    model = Sequential()
    # model.add(Embedding(20000, 128, input_length=300))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, features_num, activation='relu',input_shape=(32,features_num)))
    model.add(MaxPooling1D(pool_size=features_num))
    model.add(Dropout(0.5))
    model.add(LSTM(units))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("model fitting - simplified convolutional neural network")
    # model.summary()
    return model


def multi_CNN(features_num,filter_sizes = (3, 8),num_filters = 10,dropout_prob = (0.5, 0.8),hidden_dims = 150):
    input_shape = (32, features_num)
    model_input = Input(shape=input_shape)
    z = model_input
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(3, activation="softmax")(z)

    model = Model(model_input, model_output)
    epochs = 300
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])
    print model.summary()
    return model



def model3(features_num):
    input_shape = (60, features_num)
    model_input = Input(shape=input_shape)
    z = Dropout(dropout_prob[0])(model_input)
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=features_num)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# Function to create model, required for KerasClassifier
def create_model(features_num,hidden_dims=128, kernel_size=2,pool_size=2, use_embedding=False):
    # create model
    # shape='60,'+features_num
    # hidden_dims=50
    input_shape = (32, features_num)
    embedding_layer = Embedding(14445,
                                12,
                                input_length=32)
    sequence_input = Input(shape=(32,), dtype='int32')
    model = Sequential()

    if use_embedding==True:
        model.add(embedding_layer)
        model.add(Conv1D(activation="relu", filters=128, kernel_size=kernel_size, padding='same'))
    else:
        model.add(Conv1D(activation="relu",input_shape =input_shape, filters=32, kernel_size=kernel_size,padding='same'))
    # model.add(Conv1D(nb_filter=128,filter_length=3,activation='relu',input_shape =input_shape))  # we use max pooling:
    # model.add(Conv1D(activation="relu", filters=128, kernel_size=3))


    # we use max pooling:
    model.add(MaxPooling1D(pool_size=pool_size))

    # model.add(Conv1D(activation="relu",  filters=128, kernel_size=3))
    # model.add(Conv1D(activation="relu",  filters=128, kernel_size=3))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    # model.add(LSTM(hidden_dims))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid')) #sigmoid
    # model.add(Lambda(round,output_shape=(input_shape[0], )))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    print model.summary()
    return model

def create_categorical_model(features_num,hidden_dims=128, kernel_size=2,pool_size=2, use_embedding=False):
    # create model
    # shape='60,'+features_num
    # hidden_dims=50
    input_shape = (32, features_num)
    embedding_layer = Embedding(14445,
                                12,
                                input_length=32)
    sequence_input = Input(shape=(32,), dtype='int32')
    model = Sequential()

    if use_embedding==True:
        model.add(embedding_layer)
        model.add(Conv1D(activation="relu", filters=128, kernel_size=kernel_size, padding='same'))
    else:
        model.add(Conv1D(activation="relu",input_shape =input_shape, filters=128, kernel_size=kernel_size,padding='same'))
    # model.add(Conv1D(nb_filter=128,filter_length=3,activation='relu',input_shape =input_shape))  # we use max pooling:
    # model.add(Conv1D(activation="relu", filters=128, kernel_size=3))


    # we use max pooling:
    model.add(MaxPooling1D(pool_size=pool_size))

    # model.add(Conv1D(activation="relu",  filters=128, kernel_size=3))
    # model.add(Conv1D(activation="relu",  filters=128, kernel_size=3))
    # model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dropout(0.3))
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))
    # model.add(LSTM(hidden_dims))
    model.add(Dropout(0.5))
     # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(3))
    model.add(Activation('softmax'))
    # model.add(Lambda(round,output_shape=(input_shape[0], )))

    # Compile model
    epochs = 300
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model

def create_w2v_model(features_num,embedding_weights,hidden_dims=128, kernel_size=2,pool_size=2):
    # create model
    # shape='60,'+features_num
    # hidden_dims=50

    embedding_layer = Embedding(14445,
                                features_num,
                                weights=[embedding_weights],
                                input_length=32,trainable=True)
    model = Sequential()

    model.add(embedding_layer)
    model.add(Conv1D(activation="relu", filters=128, kernel_size=kernel_size, padding='same'))

    # we use max pooling:
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    # model.add(LSTM(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model.add(Lambda(round,output_shape=(input_shape[0], )))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print model.summary()
    return model

def model4(features_num,embedding_weights):
    filter_sizes = (3, 5,7)
    num_filters = 128
    sequence_length = 32
    # embedding_dim = 12
    dropout_prob = (0.5, 0.5)
    hidden_dims = 128
    # Building model
    # ==================================================
    #
    # graph subnet with one input and one output,
    # convolutional layers concateneted in parallel
    graph_in = Input(shape=(32, features_num))
    convs = []
    for fsz in filter_sizes:
        conv = Conv1D(filters=num_filters,
                             kernel_size=fsz,
                             activation='relu',
                             padding='same',
                             strides=1)(graph_in)
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    # if not model_variation == 'CNN-static':
    #     model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
    #                         weights=embedding_weights))
    embedding_layer = Embedding(14445,
                                features_num,
                                weights=[embedding_weights],
                                input_length=32, trainable=True)

    model.add(embedding_layer)
    # model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, features_num)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model3(features_num):
    input_shape = (60, features_num)
    model = Sequential()


    # we use max pooling:
    # model.add(MaxPooling1D(pool_size=2))

    conv_0 =Conv1D(activation="relu", input_shape=input_shape, filters=64, kernel_size=3)
    conv_1 = Conv1D(activation="relu", input_shape=input_shape, filters=64, kernel_size=5)
    conv_2 = Conv1D(activation="relu", input_shape=input_shape, filters=64, kernel_size=7)

    maxpool_0 = MaxPooling1D(pool_size=features_num - 3 + 1,  border_mode='valid')(conv_0)
    maxpool_1 = MaxPooling1D(pool_size=features_num - 5 + 1,  border_mode='valid')(conv_1)
    maxpool_2 = MaxPooling1D(pool_size=features_num - 7 + 1,  border_mode='valid')(conv_2)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
    model.add(merged_tensor)
    model.add(Flatten())
    # flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3*num_filters,))(merged_tensor)
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, activation='sigmoid'))

    # this creates a model that includes
    # model = Model(input=sequence_input, output=output)

    # checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
    #                              save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def complex_model(features_num):
    sequence_input = Input(shape=(60, features_num))
    # Convolutional block
    convs = []
    filter_sizes = [3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(sequence_input)
        l_pool = MaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    # l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    # l_pool1 = MaxPooling1D(5)(l_cov1)
    # l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    # l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_merge)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(1, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - more complex convolutional neural network")
    # model.summary()
    return model

def model2(features_num):
    input_shape = (60, features_num)
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu',input_shape =input_shape))
    model.add( MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add( MaxPooling1D(5))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add( MaxPooling1D(35))  # global max pooling
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    # model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model
