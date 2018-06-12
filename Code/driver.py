import numpy as np 
import time
import sys
from utils import get_length, loader, out_to_predict, logging, evaluate
from config import version, Config
from SiameseLSTM2 import SiameseLSTM2

if version == 1:
    from SiameseLSTM import SiameseLSTM as Agent
if version == 2:
    from SiameseLSTM2 import SiameseLSTM2 as Agent
num_training_volumes = 3

def train():
    config = Config()
    agent = Agent(config)
    Dev_loader = loader("data/" + config.dev_file + ".npz", batch_size = config.batch_size, Train = False)
    num_dev_batches = Dev_loader.max_batch
    train_losses  = []
    dev_losses = []
    lowest_loss = 1e+10
    for i in range(config.n_epoch):
        train_loss = 0.0
        total_trained_batches = 0
        for j in range(num_training_volumes):
            Train_loader = loader("data/" + config.train_file + "_vol_" + str(j) + ".npz", batch_size = config.batch_size, Train = True)
            num_train_batches = Train_loader.max_batch
            total_trained_batches += num_train_batches
            for k in range(num_train_batches):
                train_dic = Train_loader.get_batch()
                train_loss+=agent.run_train_step(train_dic)
        avg_train_loss = train_loss/total_trained_batches
        dev_loss = 0.0
        dev_preds = []
        dev_ys = []
        for k in range(num_dev_batches):
            dev_dic = Dev_loader.get_batch()
            dev_batch_out, dev_batch_loss = agent.test(dev_dic)      
            dev_pred = out_to_predict(dev_batch_out, version=2)
            dev_y = dev_dic["labels"]
            dev_ys += list(dev_y)
            dev_preds += list(dev_pred)
            dev_loss += dev_batch_loss
        precision , recall, accuracy , f1  = evaluate(np.array(dev_preds) , np.array(dev_ys))
        avg_dev_loss = dev_loss/num_dev_batches
        logging(i+1, avg_train_loss , avg_dev_loss , precision , recall, accuracy , f1)
        if avg_dev_loss < lowest_loss:
            agent.save(i+1)
            lowest_loss = avg_dev_loss
        Train_loader.reset_loader()
        Dev_loader.reset_loader()


def test():
    Test_loader = loader(config.test_file, batch_size = config.batch_size, Train=False)
    config = Config()
    agent = Agent(config)
    agent.restore()
    test_dic = Test_loader.get_batch()
    test_y = test_dic["labels"]
    test_out, test_loss = agent.test(test_dic)
    test_pred = out_to_predict(test_out)
    precision , recall, accuracy , f1  = evaluate(test_pred , test_out)
    print "Final Test Evaluation"
    print "precision = {}, recall = {}, accuracy = {}, f1 = {}".format(precision , recall, accuracy , f1)

if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == "train":
        train()
    else: 
        if  opt == "test":
            test()
        else:
            print "Please indicate either train or test"


    
