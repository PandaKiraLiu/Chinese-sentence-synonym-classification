import numpy as np
import jieba
import pickle
from sklearn.preprocessing import normalize
import random

padding_length = 40
embedding_dim = 64

def get_environment():
	# Load embeddings
	words, embeddings = pickle.load(open('./polyglot-zh.pkl', 'rb'))
	print "Emebddings shape is {}".format(embeddings.shape) 
	embeddings = normalize(embeddings)
	# Load custom dict
	jieba.load_userdict("custom_tokens.txt")
	special_terms = {}
	with open("custom_tokens.txt", "r") as fc:
		for line in fc:
			word = (line[:-1]).decode("utf-8") # Get rid of \n
			special_terms[word] = True
	cnt_special = 0
	for key in special_terms:
		special_terms[key] = cnt_special
		cnt_special += 1
	# Create word_to_idx
	word_to_idx = {}
	for i in range(len(words)):
		word_to_idx[words[i]] = i
	return words, embeddings, special_terms, word_to_idx

def get_length(sequence_batch):
	arr = []
	for sentence in sequence_batch:
		arr.append(len(sentence))
	return arr

def cleanup(s):
	for i in range(len(s) - 2):
		if (s[i] == '*') and (s[i + 1] == '*') and (s[i + 2] == '*'):
			s[i + 1] = 'DEL'
			s[i + 2] = 'DEL'
	for i in range(len(s)):
		if (len(s[i]) == 1) and (ord(s[i][0]) <= 32):
			s[i] = 'DEL'
	s = [c for c in s if c != 'DEL']
	return s

def split_sentence(sent):
	return cleanup(jieba.lcut(sent))

def is_Chinese_char(ch):
	if (ch >= u'\u4e00') and (ch <= u'\u9fff'):
		return True
	return False

# Assume v1 and v2 are both unit vectors
def cos_similarity(v1, v2):
	return np.dot(v1, v2)
def word_cos_similarity(word1, word2, word_to_idx, embeddings):
	return cos_similarity(embeddings[word_to_idx[word1]], embeddings[word_to_idx[word2]])

def map_embeddings(word_to_idx, special_terms, embeddings, sentence):
	result = []
	for word in sentence:
		try:
			result.append(embeddings[word_to_idx[word]])
		except:
			result.append(embeddings[0])
			# if word in special_terms:
			# 	tmp_embedding = np.zeros((embedding_dim, ))
			# 	tmp_embedding[special_terms[word] % embedding_dim] = 1
			# 	if special_terms[word] > embedding_dim:
			# 		tmp_embedding[embedding_dim - 1] = 1
			# 	result.append(tmp_embedding)
			# else:
			# 	result.append(embeddings[0])
	return result

def logging(epoch, train_loss ,dev_loss , precision , recall, accuracy, f1):
	"""
	==================== Epoch 0001 =================
	Train loss							   Dev loss   
		 0.222								  0.322
	Precision		Recall	 Accuracy		   F1
		 1.00		  1.00		 1.00		 1.00
	==================== Epoch 2 ====================

	"""
	print ("==================== Epoch " + str(epoch).zfill(4) + " =================")
	print ("Train loss							   Dev loss")
	print ("	 {}								  {}".format("%.3f" % train_loss , "%.3f" % dev_loss))
	print ("Precision		Recall	 Accuracy		   F1")
	print ("	 {}		  {}		 {}		 {}".format("%.2f" % precision, "%.2f" % recall, "%.2f" % accuracy, "%.2f" % f1))

def out_to_predict(y_out,version=1):
	if version == 1:
		return np.argmax(y_out,axis = 1)
	if version == 2:
		return 1*(y_out>0.5)

def evaluate(y_pred, y_true):
	y_true = np.reshape(y_true, (1, -1))
	tp = np.sum(y_true * y_pred)
	tn = np.sum((1 - y_true) * (1 - y_pred))
	fp = np.sum((1 - y_true) * y_pred)
	fn = np.sum(y_true * (1 - y_pred))
	precision = tp * 1.0 / (tp + fp)
	recall = tp * 1.0 / (tp + fn)
	accuracy = (tp + tn) * 1.0 / (tp + tn + fp + fn)
	f1 = 2 * precision * recall / (precision + recall)
	return precision , recall, accuracy , f1 

# Data loader
def padding(single_data):
	e = len(single_data)
	if e > padding_length:
		single_data = single_data[:padding_length]
	else:
		for i in range(padding_length - e):
			single_data.append(np.array([0 for k in range(embedding_dim)]))
	return single_data

class loader:
	def __init__(self, file, random = True, batch_size=128, Train=True):
		load_dict = np.load(file)
		c = load_dict["left_data"]
		l = len(c)
		left_data = np.zeros((l, padding_length, embedding_dim))
		for i in range(l):
			left_data[i] = padding(c[i])
		c = load_dict["right_data"]
		right_data = np.zeros((l, padding_length, embedding_dim))
		for i in range(l):
			right_data[i] = padding(c[i])
		self.left_data = left_data
		self.right_data = right_data
		#print (self.left_data.shape)
		self.labels = load_dict["labels"]
		self.left_lens = load_dict["left_lens"]
		self.right_lens = load_dict["right_lens"]
		self.batch_read = 0
		self.max_batch = len(self.left_data) / batch_size
		self.batch_size = batch_size
		self.random = random
		self.size = len(self.left_data)
		self.indices = np.arange(self.size)
		self.Train = Train
		np.random.shuffle(self.indices)

	def get_batch(self, idx = None):
		dic = {}
		#if self.Train:
		if idx is None:
			idx = self.batch_read
		if idx >= self.max_batch:
			return None # Index out of range
		self.batch_read += 1
		inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
		dic["X_left"] = self.left_data[inds]
		dic["X_right"] = self.right_data[inds]
		dic["labels"] = self.labels[inds]
		dic["left_lens"] = self.left_lens[inds]
		dic["right_lens"] = self.right_lens[inds]
		# else:
		# 	inds = random.sample(xrange(self.size),self.batch_size)
		# 	dic["X_left"] = padding(self.left_data[inds])
		# 	dic["X_right"] = padding(self.right_data[inds])
		# 	dic["labels"] = self.labels[inds]
		# 	dic["left_lens"] = self.left_lens[inds]
		# 	dic["right_lens"] = self.right_lens[inds]
		return dic

	def reset_loader(self):
		if self.random:
			np.random.shuffle(self.indices)
		self.batch_read = 0

