import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn 
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist







# kar nemikonad ba function ride mese tflearn, KERAS behtar ast
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# Building convolutional convnet
convnet = input_data(shape=[None, 28, 28, 1], name='input')
# http://tflearn.org/layers/conv/
# http://tflearn.org/activations/
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='mnist')
model.save('quicktest.model')
































'''


mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)  #10 class darim, one_hot mige ke [0,1,0,0,0,0,0,0,0,0,0,0,0] yani 1 ast. yeki roshan o baghie label ha khamoosh

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])  #28 by 28 ast vali safesh mikonim ke 784 bashe
y = tf.placeholder('float')
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64]) 
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output
    
    

def train_nn (x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epoch = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x , e_y = mnist.train.next_batch(batch_size)
                
                _, c = sess.run([optimizer, cost], feed_dict = {x:e_x, y:e_y})   
                epoch_loss +=c
                
            print 'epoch', epoch, 'compeleted out of', hm_epoch, 'loss', epoch_loss
        correct = tf.equal(tf.argmax(prediction , 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
         
        print 'accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})    
            







train_nn (x)


'''







































'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)  #10 class darim, one_hot mige ke [0,1,0,0,0,0,0,0,0,0,0,0,0] yani 1 ast. yeki roshan o baghie label ha khamoosh

hm_epoch = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunk = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunk, chunk_size])  #28 by 28 ast vali safesh mikonim ke 784 bashe
y = tf.placeholder('float')

def rnn(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1, chunk_size])
    x = tf.split(x, n_chunk, 0)
    
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs , states = tf.contrib.rnn.static_rnn(lstm_cell, x,dtype = tf.float32)
    
    
    out = tf.add(tf.matmul(outputs[-1], layer['weights']),layer ['biases'])
    
    return out
    
    

def train_nn (x):
    prediction = rnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x , e_y = mnist.train.next_batch(batch_size)
                
                
                e_x = e_x.reshape((batch_size, n_chunk, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict = {x:e_x, y:e_y})   
                epoch_loss +=c
                
            print 'epoch', epoch, 'compeleted out of', hm_epoch, 'loss', epoch_loss
        correct = tf.equal(tf.argmax(prediction , 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
         
        print 'accuracy', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunk,chunk_size)), y:mnist.test.labels})    
            







train_nn (x)
'''
























