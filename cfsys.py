# -*- coding: utf-8 -*-
__author__ = 'kangxun'
from math import sqrt

def sim_distance(prefs,person1,person2):
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
   #print si
    if len(si) == 0:#没有共同电影
        return 0
    #欧式距离平方
    sum_of_squares = sum([((prefs[person1][item]-prefs[person2][item])**2)
                         for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sqrt(sum_of_squares))#归一化

def sim_pearson(prefs,person1,person2):#-1~1 -1怎么处理 sort
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    n = len(si)
    if n == 0:
        return 0

    Ex = sum([prefs[person1][it] for it in si])/n
    Ey = sum([prefs[person2][it] for it in si])/n
    Ex2 = sum([prefs[person1][it] ** 2 for it in si])/n
    Ey2 = sum([prefs[person2][it] ** 2 for it in si])/n
    Exy = sum([prefs[person1][it]*prefs[person2][it] for it in si])/n
    num = Exy-Ex*Ey
    den = sqrt((Ex2-Ex**2)*(Ey2-Ey**2))
    if den == 0:
        return 0
    r = num/den
    return r

def topMatches(prefs,person,n = 5,similarity=sim_pearson):#-1怎么处理,A与 B 完全喜好相反
    scores=[(similarity(prefs,person,other),other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

critics = {'Lisa Rose':{'Lady in the Water':2.5,'Snakes on a Plane':3.5,
                        'Just My Luck':3.0,'Superman Returns':3.5,'You,Me and Dupree':2.5,
                        'The Night Listener':3.0},
           'Gene Seymour':{'Lady in the Water':3.0,'Snakes on a Plane':3.5,
                           'Just My Luck':1.5,'Superman Returns':5.0,'The Night Listener':3.0,
                           'You,Me and Dupree':3.5},
           'Michael Phillips':{'Lady in the Water':2.5,'Snakes on a Plane':3.0,
                               'Superman Returns':3.5,'The Night Listener':4.0},
           'Claudia Puig':{'Snakes on a Plane':3.5,'Just My Luck':3.0,
                           'The Night Listener':4.5,'Superman Returns':4.0,
                           'You,Me and Dupree':2.5},
           'Mick LaSalle':{'Lady in the Water':3.0,'Snakes on a Plane':4.0,
                           'Just My Luck':2.0,'Superman Returns':3.0,'The Night Listener':3.0,
                           'You,Me and Dupree':2.0},
           'Jack Matthews':{'Lady in the Water':3.0,'Snakes on a Plane':4.0,
                            'The Night Listener':3.0,'Superman Returns':5.0,'You,Me and Dupree':3.5},
           'Toby':{'Snakes on a Plane':4.5,'You,Me and Dupree':1.0,'Superman Returns':4.0}}
#print critics['Lisa Rose']['Lady in the Water']
#for item in critics['Toby']:
#    print item
#print sim_distance(critics,'Toby','Gene Seymour')
#print sim_pearson(critics,'Toby','Jack Matthews')
#for other in critics:
#    print other
#print topMatches(critics,'Toby',n = 3)

def getRecommendations(prefs,person,similarity=sim_pearson):#利用user 间相似度为用户推荐 item
    totals = {}#相似度加权的评分求和
    simsums = {}#相似度求和
    for other in prefs:#每一个除自己以外的人
        if other == person:
            continue
        sim = similarity(prefs,person,other)
        if sim <= 0:
            continue
        for item in prefs[other]:#只对自己未看过的电影评分 = 其他看过电影人评分加权(相似度)和
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item,0)#字典中未出现的 key 值,设置其 value = 0
                totals[item]+= prefs[other][item]*sim#评分*相似度
                #相似度之和
                simsums.setdefault(item,0)
                simsums[item] += sim
    #for item,total in totals.items():#以列表返回可遍历的(键, 值) 元组数组
    #    print item,total
    rankings = [(value/simsums[item],item)for item,value in totals.items()]#自己未看过电影的预测评分
    rankings.sort()
    rankings.reverse()
    return rankings

def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            #人与物品对调
            result[item][person] = prefs[person][item]
    return result

#print getRecommendations(critics,'Toby')
#print getRecommendations(critics,'Toby',similarity=sim_distance)
movies = transformPrefs(critics)
#print topMatches(movies,'Superman Returns')#相似电影
#print getRecommendations(critics,'Toby')#为 Toby 未评分电影打分
#print getRecommendations(movies,'Superman Returns')#为电影未评分个人


######################################################################
#先计算好item间相似度,用于对用户推荐,受用户变化影响小
def calculatesSimilarItems(prefs,n = 10):#对所有 item 返回相似 item及相似度 top10
    result = {}
    itemPrefs = transformPrefs(prefs)#从 person 为中心转为 Item 为中心
    c = 0
    for item in itemPrefs:#c 计算 item 数量
        c += 1
        if c % 100 == 0:print "%d / %d" % (c,len(itemPrefs))#当前 item 次序 除以 item 总数
        scores = topMatches(itemPrefs,item,n=n,similarity=sim_distance)#与当前 item 最相似的item 列表
        result[item] = scores
    return result

#itemSim = calculatesSimilarItems(critics)
#print itemSim

#利用 item 间相似度为用户推荐item
def getRecommendedItems(prefs,itemMatch,user):#分数由高到低返回user未评分电影
    userRatings=prefs[user]#当前用户
    scores = {}#相似度*评分 求和
    totalSim = {}#相似度求和
    for (item,rating) in userRatings.items():#遍历每一个当前user 评分过 item 及 rating
        for (similarity,item2) in itemMatch[item]:#遍历每一个 item获取相似 item 及相似度
            if item2 in userRatings:
                continue
            scores.setdefault(item2,0)
            scores[item2] += similarity*rating

            totalSim.setdefault(item2,0)
            totalSim[item2] += similarity
    rankings = [(score/totalSim[item],item) for item,score in scores.items()]#预测评分
    #list 里包含元组
    rankings.sort()
    rankings.reverse()
    return rankings

#print getRecommendedItems(critics,itemSim,'Toby')
def loadMovieLens(path = 'ml-100k'):
    movies = {}#id + title
    for line in open(path + '/u.item'):
        (id,title) = line.split('|')[0:2]#2?
        movies[id] = title

    prefs = {}
    for line in open(path + '/u.data'):
        (user,movieid,rating,ts) = line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]] = float(rating)#title作为字典的二级 key,而不是 id
    return prefs

def Recommendation(prefs,top = 3):
    for person in prefs:
            print getRecommendations(prefs,person)[0:top]#[1]只返回 movie title,不返回评分


prefs = loadMovieLens()
#Recommendation(prefs)
#print len(user)
#print len(set(user))

#print prefs['87']
#getRecommendations(prefs,'87')
#itemSim = calculatesSimilarItems(prefs,n = 50)
#print getRecommendedItems(prefs,itemSim,'87')[0:30]



