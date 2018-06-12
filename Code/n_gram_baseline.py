import sys
import pickle
from utils import *
sys.path.append("../")
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

readable_data_file_name = "readable_data_34000_4000.bin"
n_gram = 2

with open(readable_data_file_name) as f:
    to_load = pickle.load(f)
data = to_load["data"]
train_size = to_load["train_size"]
dev_size = to_load["dev_size"]

# Read in two sentences, return a dictionary of n-grams
def feature_extraction(sent_1, sent_2, n):
	ans = {}
	def single_sentence_feature_extraction(sent, flag):
		s = ["<S>"] + split_sentence(sent) + ["</S>"]
		l = len(s)
		for i in range(l - n + 1):
			tpl = " ".join(s[i:i + n])
			try:
				ans[tpl] += flag
			except:
				ans[tpl] = flag
	single_sentence_feature_extraction(sent_1, 1)
	single_sentence_feature_extraction(sent_2, -1)
	for key in ans:
		ans[key] = abs(ans[key])
	return ans

# Build train set
v = DictVectorizer(sparse=False)
X_train_dict = []
y_train = []
for i in range(train_size):
	X_train_dict.append(feature_extraction(data[i][0], data[i][1], n_gram))
	y_train.append(int(data[i][2]))
X_train = v.fit_transform(X_train_dict)

# Build dev set
X_dev_dict = []
y_dev = []
for i in range(train_size, train_size + dev_size):
	X_dev_dict.append(feature_extraction(data[i][0], data[i][1], n_gram))
	y_dev.append(int(data[i][2]))
X_dev = v.transform(X_dev_dict)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_dev_predict = clf.predict(X_dev)
print(classification_report(y_dev, y_dev_predict))
