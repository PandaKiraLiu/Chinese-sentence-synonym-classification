import pickle
import numpy as np
import jieba
import sys
import matplotlib.pyplot as plt
import random
import json
reload(sys)
sys.setdefaultencoding('utf-8')
from utils import *

train_size = 90000
dev_size = 10000
train_balance = False
words, embeddings, special_terms, word_to_idx = get_environment()

# Load training data
f = open("atec_nlp_sim_train.csv", "r")
data = []
count = 0
for lines in f:
	single_data = lines.split(",")
	if (single_data[3][0] != '0') and (single_data[3][0] != '1'): continue
	single_data[3] = int(single_data[3][0])
	data.append((single_data[1], single_data[2], single_data[3]))
random.shuffle(data)

# Handle the case where train_size is None: retain dev_size data, balance the rest by filling 1's
if train_size is None:
	dev_data = data[:dev_size]
	training_data = data[dev_size:]
	zero = 0
	one = 0
	for single_data in training_data:
		if single_data[2] == 0:
			zero += 1
		else:
			one += 1
	assert (zero > one)
	raZnd_idx = np.random.choice(len(training_data), zero - one, replace = False)
	for idx in rand_idx:
		tmp = random.randint(0, 1) # repeat the first sentence or the second?
		training_data.append((training_data[idx][tmp], training_data[idx][tmp], 1))
	random.shuffle(training_data)
	train_size = len(training_data)
	data = training_data + dev_data
elif train_balance:
	dev_data = data[:dev_size]
	training_data = []
	zero = 0
	one = 0
	for single_data in data[dev_size:]:
		if single_data[2] == 0:
			if zero * 2 == train_size: continue
			zero += 1
			training_data.append(single_data)
		else:
			if one * 2 == train_size: continue
			one += 1
			training_data.append(single_data)
	random.shuffle(training_data)
	data = training_data + dev_data
else:
	data = data[:train_size + dev_size]

print "Total number of data we have:", len(data)
print "Training data size: ", train_size
print "Dev data size: ", dev_size
assert (len(data) >= train_size + dev_size)

# test embedding coverage, vocab size
total = 0
miss = 0
vocab = set()
for i in range(len(data)):
	e = split_sentence(data[i][0])
	for word in e:
		total += 1
		if not (word in word_to_idx or word in special_terms): 
			miss += 1
		else:
			vocab.add(word)
print "Embedding Coverage: ", 1 - miss * 1.0 / total
print "Total number of words: ", len(vocab)

embedded_data_file_name = "processed_data_" + str(train_size) + "_" + str(dev_size)
readable_data_file_name = "readable_data_" + str(train_size) + "_" + str(dev_size) + ".bin"
training_data_file_name = "training_data_" + str(train_size)
dev_data_file_name = "dev_data_" + str(dev_size)

# Dump embedded data
for repeat in range(3): # 0 -> dump all data, 1 -> dump training data, 2 -> dump dev data

	if repeat == 0:
		dump_file_name = embedded_data_file_name
		index_range = range(len(data))
	elif repeat == 1:
		dump_file_name = training_data_file_name
		index_range = range(train_size)
	elif repeat == 2:
		dump_file_name = dev_data_file_name
		index_range = range(train_size, train_size + dev_size)

	left_data = []
	left_lens = []
	right_data = []
	right_lens = []
	labels = []
	for i in index_range:
		single_left_data = map_embeddings(word_to_idx, special_terms, embeddings, split_sentence(data[i][0]))
		left_lens.append(min(len(single_left_data), padding_length))
		left_data.append(single_left_data)
		single_right_data = map_embeddings(word_to_idx, special_terms, embeddings, split_sentence(data[i][1]))
		right_lens.append(min(len(single_right_data), padding_length))
		right_data.append(single_right_data)
		labels.append([data[i][2]])

	if repeat == 1: # dump into volumes
		data_per_vol = 30000
		l = len(left_data)
		for cc in range((l - 1) / data_per_vol + 1):
			with open("data/" + dump_file_name + "_vol_" + str(cc) + ".npz", "w") as f:
				np.savez(f, left_data=left_data[cc * data_per_vol:(cc + 1) * data_per_vol],
							left_lens=left_lens[cc * data_per_vol:(cc + 1) * data_per_vol],
							right_data=right_data[cc * data_per_vol:(cc + 1) * data_per_vol],
							right_lens=right_lens[cc * data_per_vol:(cc + 1) * data_per_vol],
							labels=labels[cc * data_per_vol:(cc + 1) * data_per_vol]
						)
	else:
		with open("data/" + dump_file_name + ".npz", "w") as f:
			np.savez(f, left_data=left_data, left_lens=left_lens,\
						right_data=right_data, right_lens=right_lens,\
						labels=labels)

# Dump human-readable data
to_dump = {
	"data" : data,
	"train_size" : train_size,
	"dev_size" : dev_size
}
with open("data/" + readable_data_file_name, "w") as f:
	pickle.dump(to_dump, f)


