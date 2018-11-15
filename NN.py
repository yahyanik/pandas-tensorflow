from __future__ import unicode_literals
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

import random
import pickle
import io 

from collections import Counter


lemmatizer = WordNetLemmatizer()
hm_lines = 10000000
def create_lexicon (pos, neg):
    lexicon = []

    for fi in [pos, neg]:
        with io.open(fi, 'r', encoding='cp437') as f:
            
            countent = f.readlines()
            
            for l in countent[:hm_lines]:
                l = l.lower()
                all_words = word_tokenize(l)
#                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
             
             
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    l2 = []
    for w in w_count:
        if 1000> w_count[w]>50:
            l2.append(w)


    print len(l2)       
    return l2


def sample_handling (sample, lexicon, classification):
    featureset = []
    with io.open(sample, 'r', encoding='cp437') as f:
        
        countents = f.readlines()
        for l in countents[: hm_lines]:
            current_word = word_tokenize(l.lower())
            current_word = [lemmatizer.lemmatize(i) for i in current_word]
            
            features = np.zeros(len(lexicon))
            for word in current_word:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] +=1
            features = list(features)
            featureset.append([features , classification])
            
            
    return featureset


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    
    lexicon = create_lexicon(pos, neg) 
    features = []
    features += sample_handling('./data/pos.txt', lexicon,[1,0])
    features += sample_handling('./data/neg.txt', lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][: -testing_size])
    train_y = list(features[:,1][: -testing_size])
    test_x = list(features[:,0][ -testing_size :])
    test_y = list(features[:,1][ -testing_size :])
    return train_x, train_y, test_x, test_y






n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_classes = 2
batch_size = 100
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('./data/pos.txt','./data/neg.txt')

x = tf.placeholder('float', [None, len(train_x[0])])  #28 by 28 ast vali safesh mikonim ke 784 bashe
y = tf.placeholder('float')

def nn_model(data):
    h_1_l = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    h_2_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    h_3_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    out_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, h_1_l['weights']) ,h_1_l['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, h_2_l['weights']) ,h_2_l['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, h_3_l['weights']) ,h_3_l['biases'])
    l3 = tf.nn.relu(l3)
    
    out = tf.add(tf.matmul(l3, out_l['weights']),out_l['biases'])
    
    return out
    
    

def train_nn (x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epoch = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epoch):
            epoch_loss = 0
            i = 0
            while i< len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                i +=batch_size
                
                _, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})   
                epoch_loss +=c
                
            print 'epoch', epoch, 'compeleted out of', hm_epoch, 'loss', epoch_loss
        correct = tf.equal(tf.argmax(prediction , 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
         
        print 'accuracy', accuracy.eval({x:test_x, y:test_y})    
            







train_nn (x)

























'''
# nn with mnist
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)  #10 class darim, one_hot mige ke [0,1,0,0,0,0,0,0,0,0,0,0,0] yani 1 ast. yeki roshan o baghie label ha khamoosh
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])  #28 by 28 ast vali safesh mikonim ke 784 bashe
y = tf.placeholder('float')

def nn_model(data):
    h_1_l = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    h_2_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    h_3_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    out_l = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, h_1_l['weights']) ,h_1_l['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, h_2_l['weights']) ,h_2_l['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, h_3_l['weights']) ,h_3_l['biases'])
    l3 = tf.nn.relu(l3)
    
    out = tf.add(tf.matmul(l3, out_l['weights']),out_l['biases'])
    
    return out
    
    

def train_nn (x):
    prediction = nn_model(x)
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






























