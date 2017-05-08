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
	#using tensorflow simple neural network for classify (overkill, lol)
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

	sess = tf.Session() #is this can be changed to tf.Session()?
	tf.global_variables_initializer().run(session = sess)

	#training process
	for epoch in range(training_epochs):
		#print (epoch);
		for i in range(total_batch-1):
			batch_x = trfeat[i*batch_size:(i+1)*batch_size]
			batch_y = trlabel[i*batch_size:(i+1)*batch_size]
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	#saving weight and bias

	print (np.count_nonzero(~np.isnan(W.eval(session = sess))))
	print (np.count_nonzero(~np.isnan(b.eval(session = sess))))
	saver = tf.train.Saver([W, b])
	saver.save(sess, './model.ckpt')
	print ('model saved!');
	#logging
	log = open("log.txt","a+");
	log.write(str(trlabel) + ' ' + 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()));
	log.close();

def next_train(trfeat,trlabel):
	#import for preparing dataset
	import pandas as pd
	import numpy as np
	import datetime
	#using tensorflow simple neural network for classify (overkill, lol)
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

	sess = tf.Session() #is this can be changed to tf.Session()?
	tf.global_variables_initializer().run(session = sess)

	saver = tf.train.Saver([W, b])
	saver.restore(sess, './model.ckpt')
	print ('model restored!')
	#training process
	for epoch in range(training_epochs):
		#print (epoch);
		for i in range(total_batch-1):
			batch_x = trfeat[i*batch_size:(i+1)*batch_size]
			batch_y = trlabel[i*batch_size:(i+1)*batch_size]
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	#saving weight and bias

	print (np.count_nonzero(~np.isnan(W.eval(session = sess))))
	print (np.count_nonzero(~np.isnan(b.eval(session = sess))))
	saver = tf.train.Saver([W, b])
	saver.save(sess, './model.ckpt')
	print ('model saved!');
	#logging
	log = open("log.txt","a+");
	log.write(str(trlabel) + ' ' + 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()));
	log.close();

def predict(trfeat):
	import pandas as pd
	import numpy as np
	import datetime
	#using tensorflow simple neural network for classify (overkill, lol)
	import tensorflow as tf
	tf.reset_default_graph()

	learning_rate = 0.5

	x = tf.placeholder(tf.float32, [None,3136])
	#y_ = tf.placeholder(tf.float32, [None, 2])
	W = tf.Variable(tf.zeros([3136, 2]))
	b = tf.Variable(tf.zeros([2]))

	y = tf.matmul(x, W) + b
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	sess = tf.Session() #is this can be changed to tf.Session()?
	tf.global_variables_initializer().run(session = sess)

	saver = tf.train.Saver([W, b])
	saver.restore(sess, './model.ckpt')
	print ('model restored!')
	#predict
	#print (trfeat)
	#print(W.eval(session = sess))
	#print (b.eval(session = sess))
	print(np.argmax(sess.run(y, feed_dict={x: trfeat})))