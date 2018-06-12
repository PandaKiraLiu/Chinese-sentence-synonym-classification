import numpy as np 
import time
from model import model
import sys
import tensorflow as tf
from tensorflow.contrib import layers as layers
from tensorflow.contrib import rnn
from utils import  get_length, loader, out_to_predict , logging

class SiameseLSTM(model):

    def add_placeholder(self):
        self.X_left = tf.placeholder(tf.float32,shape=[self.config.batch_size, None, self.config.input_dim])
        self.X_right = tf.placeholder(tf.float32,shape=[self.config.batch_size, None, self.config.input_dim])
        self.label = tf.placeholder(tf.int32,shape=[self.config.batch_size ,1])
        self.left_lens = tf.placeholder(tf.int32, shape=[self.config.batch_size])
        self.right_lens = tf.placeholder(tf.int32, shape=[self.config.batch_size])

    def build_network(self):
        with tf.variable_scope("lstm"):
            cells = [rnn.BasicLSTMCell(n,state_is_tuple=True) for n in self.config.cell_hidden_sizes ]
            self.lstm_cell =  rnn.MultiRNNCell(cells)
            self.initial_stateL = self.lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
            outputsL, self.stateL_tuple = tf.nn.dynamic_rnn(self.lstm_cell, self.X_left,
                                   initial_state=self.initial_stateL,
                                   sequence_length=self.left_lens, 
                                   dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            self.initial_stateR = self.lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
            outputsR, self.stateR_tuple = tf.nn.dynamic_rnn(self.lstm_cell, self.X_right,
                                   initial_state=self.initial_stateR,
                                   sequence_length = self.right_lens,
                                   dtype=tf.float32)
            self.stateL = tf.concat([self.stateL_tuple[1][0],self.stateL_tuple[1][1]],axis=1)
            self.stateR = tf.concat([self.stateR_tuple[1][0],self.stateR_tuple[1][1]],axis=1)
            self.state = tf.concat([self.stateL,self.stateR],axis=1)
        with tf.variable_scope("fully_connected"):
            self.l1 = layers.fully_connected(self.state,self.config.l1_dim,activation_fn = self.config.act)
            self.out = layers.fully_connected(self.l1 , self.config.out_dim ,activation_fn = self.config.act)
            


    def create_feed_dict(self, dic):
        feed_dict = {self.X_left: dic["X_left"], self.X_right: dic["X_right"], 
            self.label: dic["labels"], 
            self.left_lens: dic["left_lens"],
            self.right_lens: dic["right_lens"]
            }
        return feed_dict


    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.label,2), logits= self.out))









    
    
