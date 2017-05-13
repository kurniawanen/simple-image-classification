#Initial model for Iris species prediction

def train(trfeat,trlabel):
	import os;
	if not os.path.isfile('log.txt'):
		first_train(trfeat,trlabel);
	else:
		next_train(trfeat,trlabel);

def first_train(trfeat,trlabel):
	#import for preparing dataset
	import pandas as pd
	import numpy as np
	import datetime
	#using tensorflow simple neural network (1 [3136] weight layer + 1 [2] bias) for classify (overkill, lol)
	#read this for reference:https://www.tensorflow.org/get_started/mnist/beginners
	import tensorflow as tf
	tf.reset_default_graph()

	training_epochs = 25
	learning_rate = 0.5
	batch_size = 5
	total_batch = (len(trfeat)//batch_size);

	x = tf.placeholder(tf.float32, [None,3136])
	y_ = tf.placeholder(tf.float32, [None, 2])
	W = tf.Variable(tf.zeros([3136, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.matmul(x, W) + b
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
	sess = tf.Session()
	tf.global_variables_initializer().run(session = sess)

	#training process
	for epoch in range(training_epochs):
		for i in range(total_batch-1):
			batch_x = trfeat[i*batch_size:(i+1)*batch_size]
			batch_y = trlabel[i*batch_size:(i+1)*batch_size]
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	
	#saving weight and bias
	saver = tf.train.Saver([W, b])
	saver.save(sess, './model.ckpt')
	print ('model saved!');
	
	#logging
	log = open("log.txt","a+");
	log.write("Train:" + ' ' + 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()));
	log.close();

def next_train(trfeat,trlabel):
	#import for preparing dataset
	import pandas as pd
	import numpy as np
	import datetime
	#using tensorflow simple neural network (1 [3136] weight layer + 1 [2] bias) for classify (overkill, lol)
	#read this for reference:https://www.tensorflow.org/get_started/mnist/beginners
	import tensorflow as tf
	tf.reset_default_graph()

	training_epochs = 25
	learning_rate = 0.5
	batch_size = 5
	total_batch = (len(trfeat)//batch_size);

	x = tf.placeholder(tf.float32, [None,3136])
	y_ = tf.placeholder(tf.float32, [None, 2])
	W = tf.Variable(tf.zeros([3136, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.matmul(x, W) + b
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	sess = tf.Session()
	tf.global_variables_initializer().run(session = sess)

	#loading model (weight and bias from previous learn)
	saver = tf.train.Saver([W, b])
	saver.restore(sess, './model.ckpt')
	print ('model restored!')
	
	#training process
	for epoch in range(training_epochs):
		for i in range(total_batch-1):
			batch_x = trfeat[i*batch_size:(i+1)*batch_size]
			batch_y = trlabel[i*batch_size:(i+1)*batch_size]
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	
	#saving weight and bias
	saver = tf.train.Saver([W, b])
	saver.save(sess, './model.ckpt')
	print ('model saved!');
	
	#logging
	log = open("log.txt","a+");
	log.write("Train:" + ' ' + 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()));
	log.close();

def predict(trfeat):
	import pandas as pd
	import numpy as np
	import datetime
	import tensorflow as tf
	tf.reset_default_graph()

	x = tf.placeholder(tf.float32, [None,3136])
	W = tf.Variable(tf.zeros([3136, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.matmul(x, W) + b

	sess = tf.Session()
	tf.global_variables_initializer().run(session = sess)

	#loading the model
	saver = tf.train.Saver([W, b])
	saver.restore(sess, './model.ckpt')
	print ('model restored!')
	
	#predict (1 for yes and 0 for no)
	print(np.argmax(sess.run(y, feed_dict={x: trfeat})))