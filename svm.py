
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import csv
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit

company = "AMZN"

train_input = []
with open(company + '_input.csv', 'rb') as csvfile:
	rowreader = csv.reader(csvfile, delimiter=',')
	for row in rowreader:
		print row
		break
	for row in rowreader:
		temp_list = []
		for i in row[1:len(row)]:
			temp_list.append(float(i))
		train_input.append(np.array(temp_list))


train_output = []
with open(company + '_output.csv', 'rb') as csvfile:
	rowreader = csv.reader(csvfile, delimiter=',')
	for row in rowreader:
		print row
		break
	for row in rowreader:
		train_output.append(float(row[1]))


# NUM_EXAMPLES = int(0.8 * len(train_output)) # TODO change number of examples


# test_input = train_input[NUM_EXAMPLES:]
# test_output = train_output[NUM_EXAMPLES:]

# NUM_EXAMPLES = 300000
# train_input = train_input[:NUM_EXAMPLES]
# train_output = train_output[:NUM_EXAMPLES]

# tscv = TimeSeriesSplit(n_splits=3)
# TimeSeriesSplit(max_train_size=None, n_splits=10)
# for train, test in tscv.split(train_input):
# 	print("%s %s" % (train, test))

# exit(0)
lin_clf = svm.LinearSVC(class_weight = "balanced", multi_class="ovr")

scoring = ['f1_micro','precision_macro', 'recall_macro']
cv = cross_validate(lin_clf, train_input, train_output, scoring=scoring, cv=3, return_train_score=True)
# cv = cross_val_score(lin_clf, train_input, train_output, scoring='accuracy')
print cv


