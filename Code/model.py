import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.layers as layers





class model(object):
    def __init__(self,config):
        self.config = config
        self.sess = tf.Session()
        self.add_placeholder()
        self.build_network()
        self.add_loss()
        self.add_train()
        self.initialize()


    def add_placeholder(self):
        raise NotImplementedError("Each model must add its own placeholder!")

        
    def build_network(self):
        raise NotImplementedError("Each model must build its own network!")

    def create_feed_dict(self, dic):
        raise NotImplementedError("Each model must build its own feed dictionary!")  

    def add_loss(self):
        raise NotImplementedError("Each model must define its own loss!")  


    def add_train(self):
        if not hasattr(self, 'loss'):
            raise NotImplementedError("Loss has not been implemented!")  
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.config.lr).minimize(self.loss)

    def initialize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def run_train_step(self, dic):
        dic = self.create_feed_dict(dic)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict = dic)
        return loss

    def save(self,step):
        if not os.path.exists(self.config.model_output):
            os.mkdir(self.config.model_output)
        self.saver.save(self.sess,"./"+self.config.model_output+"/"+str(step)+".ckpt",global_step=step)


    def test(self, dic):
        if not hasattr(self,'out'):
            raise NamingError("Output should be named as self.out in build_network")
        feed_dict = self.create_feed_dict(dic)
        out, loss = self.sess.run([self.out, self.loss],feed_dict = feed_dict)
        return out, loss


    def restore(self):
        self.saver.restore(self.sess,self.config.ckpt)





