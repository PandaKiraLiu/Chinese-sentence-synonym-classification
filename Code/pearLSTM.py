import pickle
import numpy as np
import sys
import random
from utils import *
sys.path.append("../")
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
from sklearn.metrics import classification_report

class TfBidirectionalRNNClassifier(TfRNNClassifier):
    
    def build_graph(self):
        self._define_embedding()

        self.inputs = tf.placeholder(
            tf.int32, [None, self.max_length])

        self.ex_lengths = tf.placeholder(tf.int32, [None])

        # Outputs as usual:
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # This converts the inputs to a list of lists of dense vector
        # representations:
        self.feats = tf.nn.embedding_lookup(
            self.embedding, self.inputs)

        # Same cell structure as the base class, but we have
        # forward and backward versions:
        self.cell_fw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_dim, activation=self.hidden_activation)
        
        self.cell_bw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_dim, activation=self.hidden_activation)

        # Run the RNN:
        outputs, finals = tf.nn.bidirectional_dynamic_rnn(
            self.cell_fw,
            self.cell_bw,
            self.feats,
            dtype=tf.float32,
            sequence_length=self.ex_lengths)
      
        # finals is a pair of `LSTMStateTuple` objects, which are themselves
        # pairs of Tensors (x, y), where y is the output state, according to
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
        # Thus, we want the second member of these pairs:
        last_fw, last_bw = finals          
        last_fw, last_bw = last_fw[1], last_bw[1]
        
        last = tf.concat((last_fw, last_bw), axis=1)
        
        self.feat_dim = self.hidden_dim * 2               

        # Softmax classifier on the final hidden state:
        self.W_hy = self.weight_init(
            self.feat_dim, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        self.model = tf.matmul(last, self.W_hy) + self.b_y    


words, embeddings, special_terms, word_to_idx = get_environment()

# Load train and dev set
def load_data(data_list, all_vocab, train):
    X = []
    y = []

    def rua(lst):
        for i in range(len(lst)):
            if not (lst[i] in all_vocab):
                if train:
                    all_vocab[lst[i]] = True
                else:
                    lst[i] = "$UNK"
        return lst

    def feature_extraction(s1, s2):
        while True:
            flag = False
            for i in range(len(s1)):
                for j in range(len(s2)):
                    if s1[i] == s2[j]:
                        del s1[i]
                        del s2[j]
                        flag = True
                        break
                if flag: break
            if not flag: break
        return ["<S>"] + rua(s1) + ["<M>"] + rua(s2) + ["</S>"]

    for single_data in data_list:
        single_X = feature_extraction(split_sentence(single_data[0]), split_sentence(single_data[1]))
        X.append(single_X)
        y.append(int(single_data[2]))
    return X, y


############################ Load Method 1 ###########################
# readable_data_file_name = "readable_data_34000_4000.bin"

# with open(readable_data_file_name) as f:
#     to_load = pickle.load(f)
# data = to_load["data"]
# train_size = to_load["train_size"]
# dev_size = to_load["dev_size"]

# all_vocab = {"$UNK" : True}
# X_rnn_train, y_rnn_train = load_data(data[:train_size], all_vocab, train=True)
# X_rnn_dev, y_rnn_dev = load_data(data[train_size:train_size + dev_size], all_vocab, train=False)


############################ Load Method 2 ###########################
with open("data/readable_data_90000_10000.bin", "r") as f:
    data = pickle.load(f)["data"]

train_data = data[:90000]
dev_data = data[90000:]

all_vocab = {"$UNK" : True}
X_rnn_train, y_rnn_train = load_data(train_data, all_vocab, train=True)
X_rnn_dev, y_rnn_dev = load_data(dev_data, all_vocab, train=False)


tf_bidir_rnn = TfBidirectionalRNNClassifier(
    all_vocab,
    embed_dim=100,
    hidden_dim=100,
    max_length=52,
    hidden_activation=tf.nn.softmax,
    cell_class=tf.nn.rnn_cell.LSTMCell,
    train_embedding=True, # TODO
    max_iter=10, # TODO
    eta=0.01)

_ = tf_bidir_rnn.fit(X_rnn_train, y_rnn_train)

tf_rnn_dev_predictions = tf_bidir_rnn.predict(X_rnn_dev)

print
print(classification_report(y_rnn_dev, tf_rnn_dev_predictions))