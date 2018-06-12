import sys
import pickle
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import classification_report
sys.path.append("../")

readable_data_file_name = "data/readable_data_90000_10000.bin"
with open(readable_data_file_name) as f:
    to_load = pickle.load(f)
data = to_load["data"]

words, embeddings, special_terms, word_to_idx = get_environment()

# Check whether two words have similar meanings
def word_match(word1, word2):
	if (word1 in special_terms) and (not (word2 in special_terms)): return False
	if (word2 in special_terms) and (not (word1 in special_terms)): return False
	if (word1 in special_terms) and (word2 in special_terms):
		if word1 == word2:
			return True
		else:
			return False
	if (word1 in word_to_idx) and (word2 in word_to_idx):
		if word_cos_similarity(word1, word2, word_to_idx, embeddings) > 0.8:
			return True
		else:
			return False
	return False

# Apply some heuristic on the dataset
my_predicts = []
labels = []
i = 0
for single_data in data:
	i += 1
	u = cleanup(jieba.lcut(single_data[0]))
	v = cleanup(jieba.lcut(single_data[1]))
	if len(u) > len(v): u, v = v, u
	match = 0
	for word in u:
		if not (word in word_to_idx or special_terms):
			match += 1
			continue
		flag = 1
		for word2 in v:
			if word_match(word, word2):
				flag = 0
				break
		if flag == 0: match += 1
	if match * 1.0 / len(u) > 0.5:
		predict = 1
	else:
		predict = 0
	my_predicts.append(predict)
	labels.append(int(single_data[2]))

print(classification_report(labels, my_predicts))
