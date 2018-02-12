import tensorflow as tf
import numpy as np
import sklearn as sk
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

tf.logging.set_verbosity(tf.logging.INFO)
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.x1 = tf.placeholder(tf.float64, [None, sequence_length,10], name="x1")
        self.x2 = tf.placeholder(tf.float64, [None, sequence_length,6], name="x2")
        self.x3 = tf.placeholder(tf.float64, [None, sequence_length,6], name="x3")
        self.x4 = tf.placeholder(tf.float64, [None, sequence_length,6], name="x4")
        self.x5 = tf.placeholder(tf.float64, [None, sequence_length,6], name="x5")
        self.x6 = tf.placeholder(tf.float64, [None, sequence_length,1], name="x6")
        self.negScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="negScore")
        self.posScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="posScore")
        self.adeScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="adeScore")
        self.subjScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="subjScore")
        self.pposScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="pposScore")
        self.nnegScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="nnegScore")
        self.moreGoodScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="moreGoodScore")
        self.moreBadScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="moreBadScore")
        self.lessBadScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="lessBadScore")
        self.lessGoodScore= tf.placeholder(tf.float32, [None, sequence_length,1], name="lessGoodScore")
        self.wordLength = tf.placeholder(tf.float32, [None, sequence_length,1], name="wordLength")
        self.wordOrder = tf.placeholder(tf.float32, [None, sequence_length,1], name="wordOrder")
        self.wordCluster= tf.placeholder(tf.float32, [None, sequence_length,10], name="cluster")
        self.wordCluster2 = tf.placeholder(tf.float32, [None, sequence_length, 10], name="cluster2")
        self.emb1= tf.placeholder(tf.float64, [None, sequence_length], name="emb1")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.w2v = tf.placeholder(tf.float32, [None, sequence_length, 150], name="cluster2")
        # self.x1 = tf.Variable(0, name="x1", trainable=True)

        # self.mode=learn.ModeKeys.TRAIN
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)




        # X = tf.concat(2, [ self.x2, self.x3,self.x1, self.x4,self.x5 ])  #,
                          #emb5,self.embedded_chars,,self.emb5,self.emb6 emb1,emb2,emb3,emb4,emb1,emb2,emb3,emb4,self.embedded_chars shape(?, 21, 100)



        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            params = tf.Variable(
                tf.constant([0, 150, 10, 20, 30, 160, 40, 170, 50, 180, 50]),#tf.random_uniform([7, 1], -1.0, 1.0), #tf.random_uniform([20, 1], -1.0, 255),
                     name="params",trainable=True)
            pos_neg_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            ppos_pneg_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            ade_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            subj_emb = tf.Variable(
                tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            more_good_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            less_good_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            less_bad_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            more_bad_emb = tf.Variable(tf.random_uniform([20, 1], -1.0, +1.0,tf.float32))
            word_length_emb = tf.Variable(tf.random_uniform([100, 1], -1.0, +1.0,tf.float32))
            word_Order_emb = tf.Variable(tf.random_uniform([100, 1], -1.0, +1.0,tf.float32))


            # Embedding layer
            # # params = tf.constant([0, 150, 10, 20, 30, 160, 40, 170, 50, 180, 5])
            # negScoreEmbed = tf.nn.embedding_lookup(pos_neg_emb, self.negScore,
            #                                        name='negScoreEmbed')  # position from first entity embedding
            # posScoreEmbed = tf.nn.embedding_lookup(pos_neg_emb, self.posScore, name='posScoreEmbed')
            # adeScoreEmbed = tf.nn.embedding_lookup(ade_emb, self.adeScore, name='adeScoreEmbed')
            # subjScoreEmbed = tf.nn.embedding_lookup(subj_emb, self.subjScore, name='subjScoreEmbed')
            # pposScoreEmbed = tf.nn.embedding_lookup(ppos_pneg_emb, self.pposScore, name='pposScoreEmbed')
            # nnegScoreEmbed = tf.nn.embedding_lookup(ppos_pneg_emb, self.nnegScore, name='nnegScoreEmbed')
            # moreGoodScoreEmbed = tf.nn.embedding_lookup(more_good_emb, self.moreGoodScore, name='moreGoodScoreEmbed')
            # moreBadScoreEmbed = tf.nn.embedding_lookup(more_bad_emb, self.moreBadScore, name='moreBadScoreEmbed')
            # lessBadScoreEmbed = tf.nn.embedding_lookup(less_bad_emb, self.lessBadScore, name='lessBadScoreEmbed')
            # lessGoodScoreEmbed = tf.nn.embedding_lookup(less_good_emb, self.lessGoodScore, name='lessGoodScoreEmbed')
            # wordLengthEmbed = tf.nn.embedding_lookup(word_length_emb, self.wordLength, name='wordlengthEmbed')
            # wordOrderEmbed=tf.nn.embedding_lookup(word_Order_emb,self.wordOrder, name='wordOrderEmbed')
            # self.embedded_chars_expanded = tf.stop_gradient(self.embedded_chars_expanded)
            # print self.embedded_chars_expanded.get_shape()
            #
                              # ,
            # self.X = tf.concat([ negScoreEmbed, posScoreEmbed, adeScoreEmbed, subjScoreEmbed, pposScoreEmbed, nnegScoreEmbed, moreGoodScoreEmbed, moreBadScoreEmbed,lessBadScoreEmbed, lessGoodScoreEmbed, wordLengthEmbed, wordOrderEmbed], 2)#,wordOrderEmbed
            # ,
            #

            self.X = tf.concat([self.negScore, self.posScore, self.adeScore, self.subjScore, self.pposScore, self.nnegScore,
                           self.moreGoodScore, self.moreBadScore, self.lessBadScore, self.lessGoodScore,self.wordLength,self.wordOrder], 2)
            # X = tf.stack([X, ],2)
            # print stack.shape
            # self.X = tf.expand_dims(self.X, -1)
            # X = tf.to_float(X, name='ToFloat')
            clust=tf.expand_dims(self.wordCluster, -1)
            clust2 = tf.expand_dims(self.wordCluster2, -1)
            self.X = tf.expand_dims(self.X, -1)
            # self.wordCluster=tf.expand_dims(self.wordCluster, -1)
            self.X = tf.concat([self.X,clust,clust2],2) #X,
            # self.embedded_chars_expanded = tf.expand_dims(X, -1)

            # split0, split1, split2, split3, split4 = tf.split(self.w2v, num_or_size_splits=5, axis=2)
            # # print tf.shape(split0)
            # split0 = tf.expand_dims(split0, -1)
            # split1 = tf.expand_dims(split1, -1)
            # split2 = tf.expand_dims(split2, -1)
            # split3 = tf.expand_dims(split3, -1)
            # split4 = tf.expand_dims(split4, -1)
            #
            # self.X = tf.concat([ split0, split1, split2, split3, split4], 2)  #  self.X,

            print 'self.X shape'
            print self.X.shape

            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")

            # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #

            # W = tf.Variable(
            #     tf.random_uniform([vocab_size, 128], -1.0, 1.0),
            #     name="E")
            # self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # self.W_p1emb = tf.placeholder(dtype=tf.float32,shape=[2, 32], name="W_p1emb")#tf.Variable(tf.random_uniform([2, 32], 0, +1.0,seed=2000))#
            # clear_char_embedding_padding = tf.scatter_update(self.W_p1emb, [0],
            #                                                  self.x1)
            # self.emb1 = tf.stop_gradient(self.emb1)

            # emb1 = tf.nn.embedding_lookup(self.W_p1emb, self.x1,name='emb1') # position from first entity embedding
            # # self.emb1 = tf.stop_gradient(self.emb1)
            #
            # # W_p2emb = tf.Variable(tf.random_uniform([2, 32], 0, +1.0,seed=2000))
            # emb2 = tf.nn.embedding_lookup(self.W_p1emb, self.x2,name='emb2')  # position from first entity embedding
            # # W_p3emb = tf.Variable(tf.random_uniform([2, 32], -1.0, +1.0))
            # emb3 = tf.nn.embedding_lookup(self.W_p1emb, self.x3,name='emb3')  # position f
            # # W_p4emb = tf.Variable(tf.random_uniform([2, 32], -1.0, +1.0))
            # emb4 = tf.nn.embedding_lookup(self.W_p1emb, self.x4,name='emb4')  # position f
            # # W_p5emb = tf.Variable(tf.random_uniform([200, 128], -1.0, +1.0))
            # # emb5 = tf.nn.embedding_lookup(W_p5emb, self.x5)  # position f
            # # W_p6emb = tf.Variable(tf.random_uniform([2, 16], -1.0, +1.0))
            # # self.emb6 = tf.nn.embedding_lookup(W_p6emb, self.x6)  # position f
            # # ids = ops.convert_to_tensor(self.x1)

            # shape = array_ops.shape(ids)
            # embeds_flat = tf.nn.embedding_lookup(self.W_p1emb, self.x1)
            # embed_shape = array_ops.concat(0, [shape, [-1]])
            # embeds = array_ops.reshape(embeds_flat, embed_shape)
            # embeds.set_shape(ids.get_shape().concatenate(self.W_p1emb.get_shape()[1:]))
            #
            # self.embedded_chars = embeds
            # embeds.set_shape(ids.get_shape().concatenate(self.W_p1emb.get_shape()[1:]))

            # self.embedded_chars = self.emb1
            # self.embedded_chars = tf.stop_gradient(self.embedded_chars)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % 5):
                # Convolution Layer
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.X,   #X,self.embedded_chars_expanded
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # conv4 = tf.layers.conv2d(
            #         inputs=pool1,
            #     filters=64,
            #     kernel_size=[2, 2],
            #     padding="VALID",
            #         name="conv-4")
            #     # Apply nonlinearity
            # h4 = tf.nn.relu(tf.nn.bias_add(conv4, self.b4), name="relu")
            #
            #     # Convolutional Layer #2 and Pooling Layer #2
            #     # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # conv5 = tf.layers.conv2d(
            #         inputs=h4,
            #         filters=64,
            #         kernel_size=[2, 2],
            #         padding="VALID",
            #         name="conv5")
            # h5 = tf.nn.relu(tf.nn.bias_add(conv5, self.b5), name="relu")
            #
            # conv6 = tf.layers.conv2d(
            #         inputs=h5,
            #         filters=64,
            #         kernel_size=[2, 2],
            #         padding="VALID",
            #         name="conv6")
            # h6 = tf.nn.relu(tf.nn.bias_add(conv6, self.b6), name="relu")
            #
            #     # pool2 = tf.nn.max_pool(value=h2, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
            # pool2 = tf.layers.max_pooling2d(inputs=h6, padding="VALID", pool_size=[2, 2], strides=1)
            #
            # conv7 = tf.layers.conv2d(
            #         inputs=pool2,
            #     filters=64,
            #     kernel_size=[5, 3],
            #     padding="VALID",
            #         name="conv-7")
            #     # Apply nonlinearity
            # h7 = tf.nn.relu(tf.nn.bias_add(conv7, self.b7), name="relu")
            #
            #     # Convolutional Layer #2 and Pooling Layer #2
            #     # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # conv8 = tf.layers.conv2d(
            #         inputs=h7,
            #         filters=64,
            #         kernel_size=[5, 3],
            #         padding="VALID",
            #         name="conv8")
            # h8 = tf.nn.relu(tf.nn.bias_add(conv8, self.b8), name="relu")
            #
            # conv9 = tf.layers.conv2d(
            #         inputs=h8,
            #         filters=64,
            #         kernel_size=[5, 3],
            #         padding="VALID",
            #         name="conv9")
            # h10 = tf.nn.relu(tf.nn.bias_add(conv9, self.b9), name="relu")
            #
            #     # pool2 = tf.nn.max_pool(value=h2, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
            # pool3 = tf.layers.max_pooling2d(inputs=h10, padding="VALID", pool_size=[2, 2], strides=1)


                # pooled_outputs.append(pool2)


            # tf.nn.max_pool(
            # h2,
            # ksize=[1, 55, 4, 1],
            # strides=[1, 1, 1, 1],
            # padding='VALID',
            # name="pool")
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,num_filters_total])
        # print 'self.h_pool_flat shape='+self.h_pool_flat.shape
        print  tf.trainable_variables()
        for v in tf.trainable_variables():
            print v.name

        # Add dropout

        # dense = tf.layers.dense(inputs=self.h_pool_flat, units=256, activation=None)

        # Add dropout operation; 0.6 probability that element will be kept

        # batch_norm = tf.contrib.layers.batch_norm(dense,
        #                                   center=True, scale=True,
        #                                   is_training=self.is_training,
        #                                   scope='bn')
        # with tf.name_scope("highway"):
        #     self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        self.h2 = tf.contrib.layers.batch_norm(self.h_pool_flat,
                                               center=True, scale=True,
                                               is_training=self.is_training,
                                               scope='bn')

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h2, self.dropout_keep_prob)
        # relu = tf.nn.relu(tf.nn.bias_add(self.h_drop, self.b3), name="relu2")

        # logits1 = tf.layers.dense(inputs=self.h_drop, units=1)
        # tanh = tf.nn.tanh(tf.nn.bias_add(logits1, self.b4), name="tanh2")
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # tf.train.GradientDescentOptimizer(0.5).minimize(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.actual=self.input_y
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # print self.predictions.get_shape()
            # print self.input_y.get_shape()

            # Accuracy

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            # y_true = tf.argmax(self.input_y, 1)
            # self.precision= tf.placeholder(tf.float32)

            # self.precision =variable_summaries(logits1,self.input_y)
                #tf.contrib.metrics.streaming_precision( tf.argmax(logits1, 1), tf.argmax(self.input_y, 1))
            # self.precision =tf.contrib.metrics.streaming_precision(logits1, self.input_y, name="precision")
            # variable_summaries(logits1, self.input_y)
            # in_y =np.array( self.input_y)
            # print in_y
            # y_true = np.argmax(in_y, 1)
            # pred=np.array(self.scores)
            # y_pred = np.argmax(pred, 1)


            # y_pred = np.argmax(self.scores, 0)
            #		print "Precision", sk.metrics.precision_score(y_true, y_pred, average=None )
            #   		print "Recall", sk.metrics.recall_score(y_true, y_pred, average=None )
            # self.fscore= sk.metrics.f1_score(y_true, y_pred, average=None)
            # self.precision=sk.metrics.f1_score(y_true, y_pred, average=None)
            # self.precision=sk.metrics.precision_score(self.input_y, correct_predictions)
            # self.recall= sk.metrics.recall_score(self.input_y, correct_predictions)
            # self.fscore= sk.metrics.f1_score(self.input_y, correct_predictions)
def variable_summaries(logits1,input_y):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  precision=tf.contrib.metrics.streaming_precision(tf.argmax(logits1, 1), input_y) #tf.argmax(input_y, 1))
  tf.summary.scalar('precision1', precision)

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
  """Highway Network (cf. http://arxiv.org/abs/1505.00387).
  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y
  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
  """
  output = input_
  for idx in xrange(layer_size):
    output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))

    transform_gate = tf.sigmoid(
      tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
    carry_gate = 1. - transform_gate

    output = transform_gate * output + carry_gate * input_

  return output
