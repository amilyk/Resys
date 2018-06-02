__author__ = 'kangxun'

import numpy as np

# -*- coding: utf-8 -*-
def loadData(path,subpath,subpath2 = None):
    prefs = {}
    for line in open(path + subpath):
        (user,movieid,rating,ts) = line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movieid] = float(rating)

    if subpath2 is None:
        return prefs
    else:
        movies = {}#id + title
        for line in open(path + subpath2):
            (id,title) = line.split('|')[0:2]#2?
            movies[id] = title
        return prefs,movies



#def splitdata():

prefs,movies = loadData('ml-100k','/u.item')
#prefs,movies = loadData('data','/ratings.dat')
#print prefs['87']

# Split
training_ratio = 0.8
Train = {}
Test = {}

f_train = open("data/train.csv",'wb')
f_test = open("data/test.csv",'wb')

for user in prefs:
    for item in prefs[user]:
        p = np.random.rand()
        s = str(user)+","+str(item)+","+str(prefs[user][item])+"\n"
        if (p < training_ratio):
            Train.setdefault(user,{})
            Train[user][item] = prefs[user][item]
            f_train.write(s)
        else:
            Test.setdefault(user,{})
            Test[user][item] = prefs[user][item]
            f_test.write(s)

f_train.close()
f_test.close()