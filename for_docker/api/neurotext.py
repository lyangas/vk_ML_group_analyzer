# -*- coding: utf-8 -*-
import time

from flask import Flask, jsonify, request, make_response, abort
from wtforms import Form, StringField, validators, FieldList, IntegerField
from wtforms.validators import DataRequired
import wtforms_json
import json
import requests

import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import urllib
import emoji

from math import isnan
import os

from threading import Thread

            
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()                    
        self.fc1 = nn.Linear(768, hidden_size)  
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU6()               
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU6()  
        self.fc4 = nn.Linear(hidden_size, 1)
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):                              
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = self.fc4(out)
        out = self.sigm(out)
        return out

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()                    
        self.fc1 = nn.Linear(768, 512)  
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()               
        self.fc3 = nn.Linear(256, 64 )
        self.relu2 = nn.ReLU()               
        self.fc4 = nn.Linear(64, 1 )
        
        self.sigm1 = nn.Sigmoid()
        
        self.fc5 = nn.Linear(1, 64)
        self.relu3 = nn.ReLU()
        self.fc6 = nn.Linear(64, 256)
        self.relu4 = nn.ReLU()
        self.fc7 = nn.Linear(256, 512)
        self.relu5 = nn.ReLU()
        self.fc8 = nn.Linear(512, 768)
        self.sigm2 = nn.Softsign()
        
    def forward(self, x):                              
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = self.fc4(out)
        
        out = self.sigm1(out)
        
        out = self.fc5(out)
        out = self.relu3(out)
        out = self.fc6(out)
        out = self.relu4(out)
        out = self.fc7(out)
        out = self.relu5(out)
        out = self.fc8(out)
        out = self.sigm2(out)
        return out
    
    def generateText(self, x):
        out = self.fc5(x)
        out = self.relu3(out)
        out = self.fc6(out)
        out = self.relu3(out)
        out = self.fc7(out)
        out = self.relu4(out)
        out = self.fc8(out)
        out = self.sigm2(out)
        return out


class WorkerWithMembers(object):
    def apiMethod (self, method, params):
        time.sleep(0.4)
        version = '5.101'
        link = "https://api.vk.com/method/" + method + "?v=" + version
        for key in params:
            link += "&" + key + "=" + str(params[key])
        #f = urllib.request.urlopen(link)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'} 
        params.update({'v': '5.101'})
        res = requests.post("https://api.vk.com/method/" + method, data=params, headers=headers).json() #get(link).json()#json.loads(f.read())
        try:
            res['response']
        except Exception:
            raise ValueError(res['error']['error_msg'])
        return res

    def getUsersIds (self, users, access_token):
        user_ids = ''
        for user in users:
            user = json.loads(user.replace("\'", "\""))['Ссылка']
            if user_ids != '':
                user_ids += ',' + str(user)
            else:
                user_ids = str(user)
        try:
            params = {'user_ids': user_ids, 
                      'access_token': access_token}
            users = self.apiMethod('users.get', params)['response']
            res = []
            for user in users:
                res.append(user['id'])
        except Exception as e:
            raise ValueError(str(e))
        return res

    def getGroupsByUserId (self, user_id, access_token):
        params = {'user_id': user_id, 
                  'access_token': access_token}
        try:
            res = self.apiMethod('users.getSubscriptions', params)['response']
        except Exception as e:
            raise ValueError ('user id=' + str(user_id) + ' error msg: ' + str(e))
        return res

    def getCommonGroups(self, users, access_token, commonUsersFrom, commonUsersTo):
        commonUsersFrom = int(commonUsersFrom)
        commonUsersTo = int(commonUsersTo)
        userIds = self.getUsersIds(users, access_token)
        listOfGroupd = {}
        for userId in userIds:
            try:
                groupIds = self.getGroupsByUserId(userId, access_token)['groups']['items']
                for groupId in groupIds:
                    groupId = str(groupId)
                    try:
                        listOfGroupd[groupId] += 1
                    except Exception:
                        listOfGroupd[groupId] = 1
            except Exception as e:
                print (str(e))
                pass

        resultListOfGroupd = []
        
        for groupId in listOfGroupd:
            if ((listOfGroupd[groupId] >= commonUsersFrom) or (commonUsersFrom == 0)) and ((listOfGroupd[groupId] <= commonUsersTo) or (commonUsersTo == 0)):
                resultListOfGroupd.append({'Ссылка': 'https://vk.com/club' + groupId, 'Участники': listOfGroupd[groupId]})
        return resultListOfGroupd
    
    def getUsersFromGroupsURLs(self, groups, access_token, listOfUsersName):
        users = []
        ids = []
        for group in groups:
            i = 0
            while True:
                params = {'access_token': access_token,
                          'group_id': group.split('club')[1],
                          'offset': i * 1000,
                          'count': 1000}
                try:
                    newIds = self.apiMethod('groups.getMembers', params)['response']['items']
                    ids = list(set(ids + newIds))
                    i += 1
                    if len(newIds) < 1000:
                        break
                except Exception as e:
                    print('groupId=' + group.split('club')[1] + ' error msg: ' + str(e))
                    break
            self.saveResultInListsOfUsers(listOfUsersName, ids)
        return ids
    
    def saveResultInListsOfUsers(self, listOfUsersName, data):
        with open('listsOfUsers/' + listOfUsersName, 'w') as file:
            for userId in data:
                file.write(str(userId) + '\n')

    def getListsOfSubscribers(self):
        themes = []
        for file in os.listdir("listsOfUsers/"):
            if (file != ".DS_Store"):
                themes.append(file.split(".")[0])
        return themes
    
    def getListOfSubscribersByName(self, listOfUsersName):
        groupsData = []
        with open("listsOfUsers/" + listOfUsersName) as file:
            for userId in file:
                groupsData.append((userId.replace('\n', '')))
        return groupsData

class ClassifyClass (object):
    def __init__  (self):
        self.tokenizer = BertTokenizer.from_pretrained('./rubert')
        self.model = BertModel.from_pretrained('./rubert')
        self.model.eval()
        self.progress = {'classify': {}, 'learn': {}, 'downloadUsers': {}}
        self.folders = ['Посты-NN', "Название-NN"]

    def getAllThemes (self):
        themes = []
        for file in os.listdir("Посты-NN/"):
            if (file != ".DS_Store"):
                themes.append(file.split(".")[0])
        return themes
    
    def getlistOfSessions (self):
        themes = []
        for file in os.listdir("sessions/"):
            if (file != ".DS_Store"):
                themes.append(file.split(".")[0])
        return themes
    
    def getDataFromSession(self, sessionName):
        groupsData = []
        with open("sessions/" + sessionName) as file:
            for groupData in file:
                groupsData.append(json.loads(groupData))
        return groupsData
    
    def delSession(self, sessionName):
        try:
            os.remove('sessions/' + sessionName)
        except Exception:
            pass
        try:
            del self.progress['classify'][sessionName]
        except Exception:
            pass
        return sessionName
    
    def removeTheme (self, theme):
        for folder in self.folders:
            for file in os.listdir(folder + "/"):
                if (file.split(".")[0] == theme):
                    os.remove(folder + '/' + file)
        return self.getAllThemes()
    
    def apiMethod (self, method, params):
        time.sleep(0.4)
        version = '5.101'
        link = "https://api.vk.com/method/" + method + "?v=" + version
        for key in params:
            link += "&" + key + "=" + str(params[key])
        #f = urllib.request.urlopen(link)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'} 
        params.update({'v': '5.101'})
        res = requests.post("https://api.vk.com/method/" + method, data=params, headers=headers).json() #get(link).json()#json.loads(f.read())
        try:
            res['response']
        except Exception:
            print('error msg:' + res['error']['error_msg'])
            raise ValueError(res['error']['error_msg'])
        return res
    
    def saveToFile (self, data, file):
        with open(file, 'w') as outfile:
            json.dump(data.tolist(), outfile)

    def loadFromFile (self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return torch.FloatTensor(data)
    
    def deEmojify(self, inputString):
        allchars = [str for str in inputString]
        for i in range(len(allchars)):
            if allchars[i] in emoji.UNICODE_EMOJI:
                allchars[i] = ' '
        return ''.join(allchars)
    
    def text2vec (self, text):
        #text = self.deEmojify(text)
        marked_text = "[CLS] " + text + " [SEP]"

        tokenized_text = self.tokenizer.tokenize(marked_text)[:500]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        sentence_embedding = torch.mean(encoded_layers[len(encoded_layers) - 1], 1)[0]
        return sentence_embedding
    
    def textesOfGroup(self, urlGroup, countName=0, countDescription=0, countStatus=0, countPosts=0, countLinks=0, countAlbums=0, countVideos=0, countTopcis=0, access_token='', closedGroups='И закрытые и открытые'):
        if ('club' in urlGroup):
            group_id = urlGroup.split('club')[1]
        elif ('public' in urlGroup):
            group_id = urlGroup.split('public')[1]
        elif ('com/' in urlGroup):
            group_id = urlGroup.split("com/")[1]
        else:
            group_id = urlGroup
            
        try:
            params = {'group_id': group_id, 
                      'fields': "members_count,description,links,status",
                      'access_token': access_token}
            #photo_100,city,country,cover,main_album_id,place,fixed_post
            res = self.apiMethod('groups.getById', params)
            res = res['response'][0]
            
        except Exception as e:
            #print('group_id=' + str(group_id) + ':' + res['error']['error_msg'])
            print(str(e))
            raise ValueError(res['error']['error_msg'])
            
        if 'deactivated' in res:
            print('group_id=' + str(group_id) + ': is deleted or banned')
            raise ValueError('bad_group')
        if closedGroups != 'И закрытые и открытые':
            if closedGroups == 'Только закрытые':
                if res['is_closed'] != 1:
                    print('group_id=' + str(group_id) + ': is not close')
                    raise ValueError('bad_group')
            if closedGroups == 'Только открытые':
                if res['is_closed'] != 0:
                    print('group_id=' + str(group_id) + ': is not open')
                    raise ValueError('bad_group')
        
        if res['is_closed'] == 2:
            print('group_id=' + str(group_id) + ': is private')
            raise ValueError('bad_group')
                    
        ownerId = str(res['id'])
        name = res['name']
        members_count = res['members_count']
        allTextes = {'nameOfGroup': name, 'members_count': members_count, 'ownerId': ownerId}

        data = {'Название': [], 
                'Описание': [],
                'Статус': [],
                'Ссылки': [],
                'Альбомы': [],
                'Видео': [],
                'Обсуждения': [],
                'Посты': []}

        if (int(countName) > 0):
            try:
                data['Название'].append(self.deEmojify(res['name']))
            except Exception:
                pass

        if (int(countDescription) > 0) :
            try:
                data['Описание'].append(self.deEmojify(res['description']))
            except Exception:
                pass
        
        if (int(countStatus) > 0):
            try:
                data['Статус'].append(self.deEmojify(res['status']))
            except Exception:
                pass

        if (int(countLinks) > 0):
            try:
                links = res['links']
                nomberOfLinks = 0
                for i in range(len(links)):
                    if nomberOfLinks < countLinks:
                        data['Ссылки'].append(self.deEmojify(links[i]['name']))
                        nomberOfLinks += 1
                    else:
                        break
            except Exception:
                pass

        if int(countAlbums) > 0:
            try:
                params = {'owner_id': '-' + ownerId,
                          'count': countAlbums,
                          'access_token': access_token}
                res = self.apiMethod('photos.getAlbums', params)
                items = res['response']['items']
                for item in items:
                    try:
                        data['Альбомы'].append(self.deEmojify(item['title']))
                    except Exception:
                        pass
                    try:
                        data['Альбомы'].append(self.deEmojify(item['description']))
                    except Exception:
                        pass
            except Exception:
                pass

        if int(countVideos) > 0:
            try:
                params = {'owner_id': '-' + ownerId, 
                          'count': countVideos,
                          'access_token': access_token}
                res = self.apiMethod('video.get', params)
                items = res['response']['items']
                for item in items:
                    try:
                        data['Видео'].append(self.deEmojify(item['title']))
                    except Exception:
                        pass
                    try:
                        data['Видео'].append(self.deEmojify(item['description']))
                    except Exception:
                        pass

            except Exception:
                pass

        if int(countTopcis) > 0:
            try:
                params = {'group_id': ownerId, 
                          'count': countTopcis,
                          'access_token': access_token}
                res = self.apiMethod('board.getTopics', params)
                items = res['response']['items']
                for item in items:
                    try:
                        data['Обсуждения'].append(self.deEmojify(item['title']))
                    except Exception:
                        pass
            except Exception:
                pass

        if int(countPosts) > 0:
            try:
                params = {'owner_id': '-' + ownerId,
                          'extended': '0',
                          'count': countPosts,
                          'offset': '0',
                          'access_token': access_token}
                res = self.apiMethod('wall.get', params)
                items = res['response']['items']

                for item in items:
                    data['Посты'].append(self.deEmojify(item['text']))
            except Exception:
                pass
        
        allTextes.update({'data': data})
        return allTextes

    def loadModels (self):
        models = {}
        themes = []
        for folder in self.folders:
            modelsFromFolder = {}
            for file in os.listdir(folder + "/"):
                if (file != ".DS_Store"):
                    modelDL = Net(100)
                    modelDL.load_state_dict(torch.load(folder + "/" + file))
                    theme = file.split(".")[0]
                    modelsFromFolder.update({theme: modelDL})
                    if not theme in themes:
                        themes.append(theme)
            models.update({folder: modelsFromFolder})
        return models, themes
    
    def createTextexEmbeddings (self, urlGroup, countName, countDescription, countStatus, countPosts, countLinks, countAlbums, countVideos, countTopcis, access_token, closedGroups):
        groupsObjects = self.textesOfGroup(urlGroup, countName, countDescription, countStatus, countPosts, countLinks, countAlbums, countVideos, countTopcis, access_token, closedGroups)
        textData = groupsObjects['data']
        vecsData = {}
        for objectType in textData:
            vecs = []
            for text in textData[objectType]:
                if len(text) > 2:
                    vec = self.text2vec(text)
                    vecs.append(vec)
            vecsData.update({objectType: vecs})
        returnedData = {'nameOfGroup': groupsObjects['nameOfGroup'],
                        'members_count': groupsObjects['members_count'], 
                        'ownerId': groupsObjects['ownerId'], 
                        'data': vecsData}
        return returnedData
    
    def stopClassify (self, sessionName):
        try:
            self.progress['classify'][sessionName] = 'break'
        except Exception as e:
            print(str(e))
        return 
        
    def relationOfGroup (self, sessionName, groups, countName=0, countDescription=0, countStatus=0, countPosts=0, countLinks=0, countAlbums=0, countVideos=0, countTopcis=0, access_token='', closedGroups='И закрытые и открытые'):
        groupsData = []
        
        groupsLength = len(groups)
        groupsCurrentCount = 0
        self.updateStatusList(typeOfData = 'classify', objectName = sessionName, status = 0)
        for group in groups:
            if self.progress['classify'][sessionName] == 'break':
                break
            if groupsLength == 0:
                break
            try:
                group = json.loads(group.replace("\'", "\""))
            except Exception:
                groupsLength -= 1
                pass
            groupURL = group['Ссылка']
            try:
                groupsObjects = self.createTextexEmbeddings(groupURL, countName, countDescription, countStatus, countPosts, countLinks, countAlbums, countVideos, countTopcis, access_token, closedGroups)
                group['Ссылка'] = 'https://vk.com/club' + str(groupsObjects['ownerId'])
            except Exception as e:
                groupsLength -= 1
                print(str(e))
                continue

            models, themes = self.loadModels()
            group.update({'Название': groupsObjects['nameOfGroup'],
                          'Размер аудитории': groupsObjects['members_count']})
            if group['Участники'] != 'Нет':
                 usersPerSubsc = (100 * group['Участники'] / groupsObjects['members_count'])
            else:
                usersPerSubsc = 'Нет'
            group.update({'% участников': usersPerSubsc})
                
            textData = groupsObjects['data']
            relationsData = {}
            for theme in themes:
                #анализируем каждый элемент на данную тематику
                relationsOfObject = {}
                for objectName in textData:
                    #инициализируем отношение к тематикам
                    relationForTheme = {}
                    for typeOfNN in models:
                        relationForTheme.update({typeOfNN: 0})
                    #классифицируем каждый текст двумя нейронками (на данную тему)
                    for vecOfText in textData[objectName]:
                        for typeOfNN in models:
                            if len(vecOfText) > 0:
                                relationForTheme[typeOfNN] += float(models[typeOfNN][theme](vecOfText))
                    #нормируем результат
                    countOfTextes = len(textData[objectName])
                    if countOfTextes > 0:
                        for typeOfNN in models:
                            relationForTheme[typeOfNN] = relationForTheme[typeOfNN] / countOfTextes
                    relationsOfObject.update({objectName:relationForTheme})

                relationsData.update({theme: relationsOfObject})
            groupsCurrentCount += 1
            self.updateStatusList(typeOfData = 'classify', objectName = sessionName, status = int(100 * groupsCurrentCount / groupsLength))
            
            group.update({'data': relationsData})
            self.saveResultInSession(sessionName, group)
            groupsData.append(group)
        self.updateStatusList(typeOfData = 'classify', objectName = sessionName, status = 100)
        return groupsData
    
    def updateStatusList (self, typeOfData = '', objectName = '', status = ''):#userId
        try:
            if status != 100:
                try:
                    self.progress[typeOfData][objectName] = status
                except Exception:
                    self.progress[typeOfData].update({objectName: 0})
            else:
                del self.progress[typeOfData][objectName]
        except Exception as e:
            print (str(e))
        
    
    def saveResultInSession (self, sessionName, data):
        with open('sessions/' + sessionName, 'a') as outfile:
            json.dump(data, outfile)
            outfile.write('\n')
            
    def circle (self, models, textes, relation):
        NOfTextes = len(textes)
        vecsOfRelations = torch.zeros (NOfTextes, len(models))
        for i in range(NOfTextes):
            vecOfCurrenText = self.text2vec(textes[i])
            if (torch.sum(vecOfCurrenText) != 0) and not (isnan(torch.sum(vecOfCurrenText))):
                data = Variable(vecOfCurrenText.view(-1, 768))#.cuda()
                for j in range(len(models)):
                    vecsOfRelations[i][j] = models[j](data)[0]
            else:
                vecsOfRelations[i] = torch.zeros(len(models))
                NOfTextes = NOfTextes - 1
        relation += torch.sum(vecsOfRelations, 0)
        return relation
                
    def stoppingLearn (self, theme):
        self.progress['learn'][theme] = 'stopping'
        del self.progress['learn'][theme]

    def appendDataset (self, vecs):
        dataLength = len(vecs)
        gen = Gen()
        gen.load_state_dict(torch.load("gen.pkl"))
        bedrockVec = torch.zeros(dataLength * 10, 768)
        with torch.no_grad():
            for i in range(len(bedrockVec)):
                bedrockVec[i] = gen.generateText(torch.rand(1))

        lenOfPosts = dataLength + bedrockVec.shape[0]

        label = torch.zeros(lenOfPosts)
        data = torch.zeros(lenOfPosts, 768)
        for i in range(lenOfPosts):
            if (i < dataLength):
                data[i] = vecs[i]
                label[i] = 1
            else:
                data[i] = bedrockVec[i - lenOfPosts]
        return [data, label]
    
    def train (self, groupURLs, count, theme, num_epochs=250, learning_rate=0.0001, hiddenLayer=100, access_token=''):
        try:
            dataset = self.createDataset(groupURLs, count, "1", theme, access_token)
            for folder in self.folders:
                net = Net(hiddenLayer)
                #net.cuda()
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                for epoch in range(int(num_epochs)):
                    textVec = Variable(dataset[folder][0])#.cuda())         
                    labels = Variable(dataset[folder][1])#.cuda())

                    optimizer.zero_grad()                             
                    outputs = net(textVec)
                    loss = criterion(outputs, labels.view(-1, 1))                 
                    loss.backward()                                   
                    optimizer.step()  
                torch.save(net.state_dict(), folder + '/' + theme + '.pkl')
        except Exception as e:
            print(str(e))
            pass
        del self.progress['learn'][theme]
        return self.getAllThemes()
    
    def createDataset (self, groupURLs, count, offset, theme, access_token):
        self.progress['learn'].update({theme: 0})
        postsVecs = []
        nemesVecs = []
        ind = 0
        for groupURL in groupURLs:
            if (self.progress['learn'][theme] != 'stopping'):
                try:
                    self.progress['learn'][theme] = "learning(Posts): " + str(int(100 * ind / len(groupURLs)))
                    ind += 1
                    if ('club' in groupURL):
                        group_id = groupURL.split('club')[1]
                    elif ('public' in groupURL):
                        group_id = groupURL.split('public')[1]
                    elif ('com/' in groupURL):
                        group_id = groupURL.split("com/")[1]
                    else:
                        group_id = groupURL
                        
                    try:
                        params = {'group_id': group_id, 
                                  'fields': "",
                                  'access_token': access_token}
                        res = self.apiMethod('groups.getById', params)
                        idAndName = res['response'][0]
                    except Exception as e:
                        print(str(e))
                        raise ValueError(res['error']['error_msg'])

                    if 'deactivated' in idAndName:
                        print('group_id=' + str(group_id) + ': is deleted or banned')
                        raise ValueError('bad_group')
                    if idAndName['is_closed'] != 0:
                        print('group_id=' + str(group_id) + ': is closed or private')
                        raise ValueError('bad_group')
                    
                    time.sleep(0.4)
                    ownerID = str(idAndName['id'])
                    vecOfName = self.text2vec(idAndName['name'])
                    if (torch.sum(vecOfName) != 0) and not (isnan(torch.sum(vecOfName))):
                        nemesVecs.append(vecOfName)
                    arrOfTextes = self.getArrOfTextsFromVK (ownerID, count, offset, access_token)
                    time.sleep(0.4)
                    for text in arrOfTextes:
                        vecOfCurrenText = self.text2vec(text)
                        if (torch.sum(vecOfCurrenText) != 0) and not (isnan(torch.sum(vecOfCurrenText))):
                            postsVecs.append(vecOfCurrenText)
                except Exception as e:
                    print(str(e))
                    pass
            else:
                raise ValueError('user stopped training')

        if (len(postsVecs) > 0) and (len(nemesVecs) > 0):
            datasetForPosts = self.appendDataset(postsVecs)
            datasetForNames = self.appendDataset(nemesVecs)
        else:
            raise ValueError('all groups is bad')
        return {'Посты-NN': datasetForPosts, 'Название-NN': datasetForNames}
    
    def getArrOfTextsFromVK (self, ownerId, countPosts, offset, access_token):
        params = {'owner_id': '-' + ownerId,
                  'extended': '0',
                  'count': countPosts,
                  'offset': '0',
                  'access_token': access_token}
        res = self.apiMethod('wall.get', params)
        arrOfTexts = []
        try:
            if (res['response']['count'] == 0):
                raise ('response.count = 0')
            for item in res['response']['items']:
                if (item['text'] != '') and (item['text'] != ' '):
                    arrOfTexts.append(self.deEmojify(item['text']).replace('\n', '')) 
        except Exception as e:
            arrOfTexts = []
        return arrOfTexts
    
    
class MainWorker(object):
    def getUserIDVKByCode(self, code):
        try:
            url = 'https://oauth.vk.com/access_token?client_id=7211908&client_secret=GQfgkrX6XJrHzkhjdONn&redirect_uri=http://ml.vtargete.ru/group_analyzer/table.html&code=' + code
            f = urllib.request.urlopen(url)
            res = json.loads(f.read())
            return res
        except Exception as e:
            print(str(e))
            #raise ValueError(str(e))
            
#----------------------------------------------------------------------------------------------------------
class RequestFormForAuthVK(Form):
    code = StringField('code', validators = [DataRequired()])

class RequestFormForGetDataFromSession(Form):
    sessionName = StringField('sessionName', validators = [DataRequired()])
    
class RequestFormForDelSessions(Form):
    sessionName = StringField('sessionName', validators = [DataRequired()])

class RequestFormForStopClassify(Form):
    sessionName = StringField('sessionName', validators = [DataRequired()])
    
class RequestFormForRemoveTheme(Form):
    theme = StringField('theme', validators = [DataRequired()])

class RequestFormForLearn(Form):
    theme = StringField('theme', validators = [DataRequired()])
    epoch = StringField('epoch', validators = [DataRequired()])
    countPosts = StringField('countPosts', validators = [DataRequired()])
    groupURLs = FieldList(StringField('groupURLs', validators = [DataRequired()]), validators = [DataRequired()])
    token = StringField('token', validators = [DataRequired()])

class RequestFormForClassifyAllThemes(Form):
    sessionName = StringField('sessionName', validators = [DataRequired()])
    URLs = FieldList(StringField('URLs', validators = [DataRequired()]), validators = [DataRequired()])
    countName = StringField('countName', validators = [DataRequired()])
    countDescription = StringField('countDescription', validators = [DataRequired()])
    countStatus = StringField('countStatus', validators = [DataRequired()])
    countPosts = StringField('countPosts', validators = [DataRequired()])
    countLinks = StringField('countLinks', validators = [DataRequired()])
    countAlbums = StringField('countAlbums', validators = [DataRequired()])
    countVideos = StringField('countVideos', validators = [DataRequired()])
    countTopcis = StringField('countTopcis', validators = [DataRequired()])
    commonUsersFrom = StringField('commonUsersFrom', validators = [DataRequired()])
    commonUsersTo = StringField('commonUsersTo', validators = [DataRequired()])
    typeOfInputData = StringField('typeOfInputData', validators = [DataRequired()])
    token = StringField('token', validators = [DataRequired()])
    closedGroups = StringField('closedGroups', validators = [DataRequired()])

class RequestFormForStoppingLearn(Form):
    theme = StringField('theme', validators = [DataRequired()])
    
class RequestFormForGetUsersFromGroupsURLs(Form):
    groupURLs = FieldList(StringField('groupURLs', validators = [DataRequired()]), validators = [DataRequired()])
    token = StringField('token', validators = [DataRequired()])
    listOfUsersName = StringField('listOfUsersName', validators = [DataRequired()])
    
class RequestFormForGetListOfSubscribersByName(Form):
    listOfSubscribersName = StringField('listOfSubscribersName', validators = [DataRequired()])
#-------------------------------------------------------------------------------------------------------------------
wtforms_json.init()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

workerWithUsers = WorkerWithMembers()
classifyClass = ClassifyClass()
mainWorker = MainWorker()
#-------------------------------------------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'unknown method'}), 404)

@app.route('/api/getStateOfProgres', methods=['GET'])
def task_getStateOfProgress():
    return jsonify(classifyClass.progress)
    
@app.route('/api/getListOfThemes', methods=['GET'])
def task_getListOfThemes():
    return jsonify(classifyClass.getAllThemes())

@app.route('/api/getlistOfSessions', methods=['GET'])
def task_getlistOfSessions():
    return jsonify(classifyClass.getlistOfSessions())

@app.route('/api/authVK', methods=['POST'])
def task_authVK():
    requestForm = RequestFormForAuthVK.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(mainWorker.getUserIDVKByCode(requestForm.code.data)), 201
    else:
        return jsonify(requestForm.errors)
    
@app.route('/api/delSessions', methods=['POST'])
def task_delSessions():
    requestForm = RequestFormForDelSessions.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.delSession(requestForm.sessionName.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/stopClassify', methods=['POST'])
def task_stopClassify():
    requestForm = RequestFormForStopClassify.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.stopClassify(requestForm.sessionName.data)), 201
    else:
        return jsonify(requestForm.errors)
    
@app.route('/api/getListsOfSubscribers', methods=['GET'])
def task_getListsOfSubscribers():
    return jsonify(workerWithUsers.getListsOfSubscribers())

@app.route('/api/getListOfSubscribersByName', methods=['POST'])
def task_getListOfSubscribersByName():
    requestForm = RequestFormForGetListOfSubscribersByName.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(workerWithUsers.getListOfSubscribersByName(requestForm.listOfSubscribersName.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/getDataFromSession', methods=['POST'])
def task_getDataFromSession():
    requestForm = RequestFormForGetDataFromSession.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.getDataFromSession(requestForm.sessionName.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/getUsersFromURLs', methods=['POST'])
def task_getUsersFromURLs():
    requestForm = RequestFormForGetUsersFromGroupsURLs.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(workerWithUsers.getUsersFromGroupsURLs(requestForm.groupURLs.data, requestForm.token.data, requestForm.listOfUsersName.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/removeTheme', methods=['POST'])
def task_removeTheme():
    requestForm = RequestFormForRemoveTheme.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.removeTheme(requestForm.theme.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/classifyAllThemes', methods=['POST'])
def task_classify_all_themes():
    requestForm = RequestFormForClassifyAllThemes.from_json(request.json)
    if (requestForm.validate()):
        groupURLs = ''
        if requestForm.typeOfInputData.data == 'Список сообществ':
            groupURLs = requestForm.URLs.data
        elif requestForm.typeOfInputData.data == 'Список пользователей':
            classifyClass.progress['classify'].update({requestForm.sessionName.data: 'парсим пользователей'})
            with open('sessions/' + requestForm.sessionName.data, 'w') as f:
                print('parsing')
            groupURLs = workerWithUsers.getCommonGroups(requestForm.URLs.data, requestForm.token.data, requestForm.commonUsersFrom.data, requestForm.commonUsersTo.data)
        else:
            print('bad type of input data: ' + requestForm.typeOfInputData.data)
            return 0
        return jsonify(classifyClass.relationOfGroup(requestForm.sessionName.data, groupURLs, requestForm.countName.data,
                                                     requestForm.countDescription.data, requestForm.countStatus.data, 
                                                     requestForm.countPosts.data, requestForm.countLinks.data,
                                                     requestForm.countAlbums.data, requestForm.countVideos.data, 
                                                     requestForm.countTopcis.data, requestForm.token.data, 
                                                     requestForm.closedGroups.data)), 201
    else:
        return jsonify(requestForm.errors)
    
@app.route('/api/learn', methods=['POST'])
def task_learn():
    requestForm = RequestFormForLearn.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.train(requestForm.groupURLs.data, requestForm.countPosts.data, requestForm.theme.data, requestForm.epoch.data, 0.0001, 100, requestForm.token.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/stoppingLearn', methods=['POST'])
def task_stopping():
    requestForm = RequestFormForStoppingLearn.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(classifyClass.stoppingLearn(requestForm.theme.data)), 201
    else:
        return jsonify(requestForm.errors)

@app.route('/api/getUsersFromGroupsURLs', methods=['POST'])
def task_getUsersFromGroupsURLs():
    requestForm = RequestFormForGetUsersFromGroupsURLs.from_json(request.json)
    if (requestForm.validate()):
        return jsonify(workerWithUsers.getUsersFromGroupsURLs(requestForm.groupURLs.data, requestForm.token.data)), 201
    else:
        return jsonify(requestForm.errors)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5080)
