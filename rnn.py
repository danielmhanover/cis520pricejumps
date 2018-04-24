#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import csv
from sklearn.metrics import confusion_matrix as confusion_matrix

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
		lol = int(row[1])
		if lol == 0:
			train_output.append(np.array([0,1,0]))
		if lol == -1:
			train_output.append(np.array([1,0,0]))
		if lol == 1:
			train_output.append(np.array([0,0,1]))

seq_length = 10

train_input = [train_input[i:i + seq_length] for i in xrange(0, len(train_input), seq_length)]
train_input = train_input[:len(train_input) - 1]
train_output = [train_output[i] for i in xrange(seq_length + 1, len(train_output), seq_length)]

NUM_EXAMPLES = int(0.8 * len(train_output)) # TODO change number of examples


test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print "test and training data loaded"


data = tf.placeholder(tf.float32, [None, seq_length,len(train_input[0][0])]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 3])
num_hidden = 15  #TODO CHANGE NUMBER OF HIDDEN
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 100 # TODOchange
no_of_batches = int(len(train_input)) / batch_size
epoch = 200
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch ",str(i)

incorrect = sess.run(error,{data: test_input, target: test_output})

# y = [[0,0,0],[0,0,0],[0,0,0]]
# for i in xrange(0, len(y_pred)):
# 	predicted = y_pred[i]
# 	if np.array_equal(test_output[i], np.array([1, 0, 0])):
# 		trueval = 0
# 	elif np.array_equal(test_output[i], np.array([0, 1, 0])):
# 		trueval = 1
# 	else:
# 		trueval = 2
# 	y[predicted][trueval] = y[predicted][trueval] + 1

# print sess.run(prediction,{data: [[[2206400,60,2205300,100],
	# [2206400,60,2205200,100],
	# [2206300,100,2205200,100],
	# [2206400,160,2205200,100],
	# [2206400,160,2205100,400],
	# [2205300,100,2205100,400],
	# [2205300,91,2205100,400],
	# [2206100,100,2205100,400],
	# [2206100,100,2205100,309],
	# [2206200,100,2205100,309]]]})

print('Epoch {:2d} microF1 precision {:3.1f}%'.format(i + 1, 100 * (1-incorrect)))
sess.close()