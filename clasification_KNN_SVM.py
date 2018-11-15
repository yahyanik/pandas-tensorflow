from __future__ import division


import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from sklearn.cluster import KMeans
import pickle
import math 
import matplotlib.pyplot as plt
import random
import warnings
from matplotlib import style
from collections import Counter




style.use('fivethirtyeight')

'''

#Ttytanic kmeans

df = pd.read_excel('./data/titanic.xls')

  
df.drop(['body','name'],1,inplace = True)
 
df.convert_objects(convert_numeric = True)
df.fillna(0,inplace = True)
 
def handle_non_numerical_data (df):
     
    columns = df.columns.values
     
    for column in columns:
        text_digit_vals = {}
        def convert_to_int (val):
            return text_digit_vals[val]
     
 
        if df[column].dtype != np.int64 and df[column].dtype!=np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
         
            df[column] = list(map(convert_to_int, df[column]))
         
    return df
 
df = handle_non_numerical_data (df)
df.drop(['boat', 'sex'],1, inplace = True)
x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = KMeans(n_clusters = 2)
clf.fit(x)
correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1

print (correct/len(x))

'''

















'''

#KMEANS
x= np.array ([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

# clf = KMeans(n_clusters=2)
# 
# clf.fit(x)
# centeroids = clf.cluster_centers_
# labels = clf.labels_

colors = 20*['g','r','b','o','k','c']
# for i in range (len(x)):
#     plt.plot(x[i][0],x[i][1], colors [labels[i]], markersize = 10)




# plt.scatter(centeroids[:,0],centeroids[:,1], s = 150, linewidths=5)
# plt.show()

#k means needs the number of the classes but MEAN SHIFT does not require that and finds the point itself.
class K_means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                
            optimation = True
            for c in self. centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) >self.tol:
                    optimization = False
            if optimization:
                break 
                
                
                
    def predict (self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification             
    
clf = K_means()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker = 'o', color = 'k', linewidths=5)
    
for classification in clf. classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker = 'x', color = color, s = 150, linewidths= 5)
        
        
        
un = np.array([[1,3],[8,9],[0,3],[5,4],[6,4]])

for un in un:
    classification = clf.predict(un)
    plt.scatter(un[0],un[1],marker = '*', color = colors[classification], s= 150, linewidths=5)
        
plt.show()
    

'''









































'''
#svm implementation


class svm:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization :
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            
    def fit (self,data):
        
        self.data = data
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        
        all_data = []
        for yi in self.data :
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min (all_data)
        
        step_size = [self.max_feature_value * 0.1,self.max_feature_value * 0.01,self.max_feature_value * 0.001]
        b_range_multiple = 5
        
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        #we dont need to take small steps for b as we do for w
        for step in step_size:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            
            while not optimized:
                
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for Transformation in transforms:
                        w_t = w*Transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b] #magnetude of a vector
                                
                if w[0] <0:
                    optimized = True
                    print 'optimized a step'
                else:
                    w = w-step        
            norms = sorted([n for n in opt_dict])
            opt_choise = opt_dict[norms[0]]   
            self.w = opt_choise[0]
            self.b = opt_choise[1]  
            latest_optimum = opt_choise[0][0]+step*2            
                
        
        
      
    
    
    def predict (self,features):
        
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,marker = '*',c = self.colors[classification])
        
        return classification
    
    
    
    def visualize (self):
        
        
        [[self.ax.scatter(x[0],x[1],s = 100,color = self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        def hyperplane (x,w,b,v):
            k = (-w[0]*x-b+v)/w[1]
            return k
        
    
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')
        
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')
        
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1, db2], 'y--')
        print self.w
        
        
        
        plt.show()
            
        
        
        

        
        
data_dict = {-1:np.array([[1,7],[2,8],[3,8]]) ,1: np.array([[5,1],[6,-1],[7,3]])}


svm = svm()
svm.fit(data = data_dict)
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]
for p in predict_us:
    svm.predict(p)
    
svm.visualize()


'''









































'''
#SVM
df = pd.read_csv('./data/breast-cancer-wisconsin.data')


df.replace('?', -99999, inplace= True)

df.drop(['id'],1,inplace = True)



x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)


clf = svm.SVC()
clf.fit(x_train, y_train)


accuracy = clf.score(x_test,y_test)

print accuracy

with open ('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open ('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

example_measure = np.array([[4,2,1,1,1,2,3,2,1]])
print example_measure
example_measure = example_measure.reshape(len(example_measure),-1) #chandta eleman darim mishe addade avali
print example_measure
prediction = clf.predict(example_measure)
print (prediction)

'''































'''
#KNN_implementation

dataset = {'k':[[1,2],[2,3],[3,1]] , 'r':[[5,6],[7,7],[8,6]]}
new_features = [5,7]


#one line code for this can be as follows:
#[[plt.scatter(ii[0], ii[1]], s=100, color = i) for ii in dataset[i]]for i in dataset]

def knn(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('the dataset length is smaller than the value needed for KNN algorithm')
     
    distances = []
    for group in data:
        for features in data[group]:
            u_d = np.linalg.norm(np.array(features)- np.array(predict)) 
            distances.append([u_d, group])
            
    
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0] 
    confidence = Counter(votes).most_common(1)[0][1]/k
    print Counter(votes).most_common(1)
    return vote_result, confidence


result = knn(dataset, new_features, k =3)
print result


# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color = i)
#         
# plt.scatter(new_features[0],new_features[1],color = result)
# plt.show()



df = pd.read_csv('./data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace= True)
df.drop(['id'],1,inplace = True)
full_data = df.astype (float).values.tolist()

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)) :]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
    
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidance = knn(train_set,data, k =5)
        if group == vote:
            correct+=1
        total +=1


accuracy = correct/total
print accuracy
'''












'''
# machine knn
df = pd.read_csv('./data/breast-cancer-wisconsin.data')


df.replace('?', -99999, inplace= True)

df.drop(['id'],1,inplace = True)



x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)


clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)


accuracy = clf.score(x_test,y_test)

print accuracy

with open ('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open ('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

example_measure = np.array([4,2,1,1,1,2,3,2,1])
example_measure = example_measure.reshape(len(example_measure),-1) #chandta eleman darim mishe addade avali
prediction = clf.predict(example_measure)
print (prediction)

'''

