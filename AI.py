from __future__ import division
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import cv2
import os
from random import shuffle
import tensorflow as tf
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
'''

tf.reset_default_graph()
train_dir = './data/train'
test_dir = './data/test'
img_size = 50
lr = 1e-3

model_name = 'dodvscat-{}-{}.model'.format(lr, '6conv-basic-video')


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        
        path = os.path.join(train_dir , img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(img_size , img_size))
        training_data.append([np.array(img), np.array(label)])
    
    shuffle(training_data)
    np.save('training_data.npy' , training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split('.') [0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        testing_data. append([np.array(img), img_num])
        
    np.save('test_data.npy', testing_data)
    return testing_data



#train_data = create_train_data() #after one time runniong this, we dont need this anymore
train_data = np.load('training_data.npy')
 
 
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')
 
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
 
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
 
model = tflearn.DNN(convnet, tensorboard_dir = 'log')
 
if os.path.exists('./{}.meta'.format(model_name)):
    model.load(model_name)
    print 'model loaded!'
     
train = train_data[: -500]
test = train_data [-500 :]
x = np.array([i[0]for i in train]).reshape(-1,img_size, img_size, 1)
y = [i[1] for i in train]
print len(x)
print len(y)
 
test_x = np.array([i[0]for i in test]).reshape(-1,img_size, img_size, 1)
test_y = [i[1] for i in test]
print (len(test_x))
print len(test_y)

# model.fit({'input': x},{'targets': y},n_epoch=5,validation_set=({'input': test_x},{'targets': test_y}),snapshot_step=700, show_metric=True, run_id=model_name)
# 
# model.save(model_name)


#test_data = process_test_data()  # if you dont have it, other wise say 
test_data = np.load('test_data.npy')

fig = plt.figure()
for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    y= fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label = 'Dog'
    else : str_label = 'Cat'
    
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()



with open ('submission.csv', 'w') as f:
    f.write('id,label\n')

with open ('submission.csv', 'a') as f:
    for data in tqdm (test_data):
        img_num = data[1]
        img_data = data[0]
        y= fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(img_size, img_size, 1)
        model_out = model.predict([data])[0]
    
        f.write('{},{}\n'.format(img_num, model_out[1]))




'''


























'''
train_dir = './data/train'
test_dir = './data/test'
img_size = 50
lr = 1e-3

model_name = 'dodvscat-{}-{}.model'.format(lr, '2conv-basic-video')


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        
        path = os.path.join(train_dir , img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(img_size , img_size))
        training_data.append([np.array(img), np.array(label)])
    
    shuffle(training_data)
    np.save('training_data.npy' , training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(testing_data)
        img_num = img.split('.', [0])
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        testing_data. append([np.array(img), img_num])
        
    np.save('test_data.npy', testing_data)
    return testing_data



train_data = create_train_data() #after one time runniong this, we dont need this anymore
#train_data = np.load('training_data.npy')


convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')

if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print 'model loaded!'
    
train = train_data[: -500]
test = train_data [-500 :]
x = np.array([i[0]for i in train]).reshape(-1,img_size, img_size, 1)
y = [i[1] for i in train]
print len(x)
print len(y)

test_x = np.array([i[0]for i in test]).reshape(-1,img_size, img_size, 1)
test_y = [i[1] for i in test]
print (len(test_x))
print len(test_y)

model.fit({'input': x},{'targets': y},n_epoch=5,validation_set=({'input': test_x},{'targets': test_y}),snapshot_step=700, show_metric=True, run_id=model_name)


'''



























lr = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_step = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_step):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
            
#some_random_games_first()         
def initial_populaion():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range (initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range (goal_step):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score +=reward
            if done :
                break
                
           
        if score >= score_requirement:
            
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)

    
    training_data_save = np.array(training_data)
    np.save('saved.npy' , training_data_save)
    
    print 'avg. accepted score:', mean(accepted_scores)
    print 'meadian accepted scores:', median(accepted_scores)
    
    return training_data




 
def nn_model(input_size):
    network = input_data(shape=[None, input_size, 1], name = 'input')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network , 2, activation = 'softmax')
     
    network = regression(network, optimizer ='adam', learning_rate= lr, loss = 'categorical_crossentropy', name = 'targets')
     
    model = tflearn.DNN(network, tensorboard_dir = 'log')
     
    return model
 
def train_model(training_data, model = False):
     
    x = np. array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
     
    if not model :
        model = nn_model(input_size = len(x[0]))
         
    model.fit({'input' : x}, {'targets' :y}, n_epoch =3 , snapshot_step = 500, show_metric = True, run_id = 'openaistuff')
     
    return model
 
# training_data = initial_populaion()
# model = train_model(training_data)
# 
# model.save ('ADFASTA.model')


model = nn_model(input_size = 4)
model.load ('ADFASTA.model')
scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range (goal_step):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            print prev_obs
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        
        
        choices.append(action)
        new_obs , reward, done, info = env.step(action)
        prev_obs = new_obs
        game_memory.append([new_obs, action])
        score+=reward
        if done:
            break
    scores.append(score)
print 'avg.score', sum (scores)/len(scores)
print 'choice1: {} , choise0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices))
   
        
     
        











