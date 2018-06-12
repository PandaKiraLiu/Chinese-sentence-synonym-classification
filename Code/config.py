import tensorflow as tf

version = 2
class Config(object):
    def act(self, state):
        return tf.nn.tanh(state)  
    ckpt =  "10000.ckpt-10000"
    model_output = "model_output"
    n_epoch = 1000
    lr = 1e-2
    input_dim = 64
    l1_dim = 8
    out_dim = 2
    cell_hidden_sizes = [32, 16]
    batch_size = 256
    train_file = "training_data_90000"
    dev_file = "dev_data_10000"
    order = 1
