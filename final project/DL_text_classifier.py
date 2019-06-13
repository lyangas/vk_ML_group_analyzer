from flask import Flask, jsonify, request, make_response, abort
from ufal.udpipe import Model, Pipeline
import torch
import torch.nn as nn
import re
import json

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):                              
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigm(out)
        return out
    
class DLTextClassifier (object):

    def __init__ (self, sourceForInf, sourceForDict, sourceForDL):
        self.nomberOfKnownWords = 0
        self.modelForInfinitive = Model.load(sourceForInf)
        self.vectorOfStr = torch.zeros(2000)
        self.sourceForDL = sourceForDL
        self.net = Net(2000, 1000, 42)
        self.net.load_state_dict(torch.load(sourceForDL, map_location='cpu'))
        self.needToSearch = False
        
        self.clasterOfWord = {}
        with open(sourceForDict) as json_file:  
            self.clasterOfWord = json.load(json_file)
        
        self.idOfThemes = {'авто/мото': 0,'активный отдых': 1,'бизнес': 2,'домашние животные': 3,'здоровье': 4,
                           'знакомство и общение': 5,'игры': 6,'ИТ (компьютеры и софт)': 7,'кино': 8,
                           'красота и мода': 9,'кулинария': 10,'культура и искусство': 11,'литература': 12,
                           'мобильная связь и интернет': 13,'музыка': 14,'наука и техника': 15,'недвижимость': 16,
                           'новости и СМИ': 17,'безопасность':18,'образование':19,'обустройство и ремонт': 20,
                           'политика': 21,'продукты питания': 22,'промышленность': 23,'путешествия':24,
                           'работа': 25,'развлечения': 26,'религия': 27,'дом и семья': 28,'спорт': 29,
                           'страхование': 30,'телевидение': 31,'товары и услуги': 32,'увлечения и хобби': 33,
                           'финансы': 34,'фото': 35,'эзотерика': 36,'электроника и бытовая техника': 37,
                           'эротика': 38,'юмор': 39,'общество, гуманитарные науки': 40,'дизайн и графика': 41}
    
    def tryToUpgradeSelfVec (self, word):
        try:
            relationID = self.clasterOfWord[word.lower()]
            self.vectorOfStr[relationID] += 1
            self.nomberOfKnownWords += 1
        except Exception:
            if not re.search(r'[\W]', word):
                relationID = self.clasterOfWord[self.wordToInf(text=word)]
                self.vectorOfStr[relationID] += 1
                self.nomberOfKnownWords += 1
            else: 
                raise
    
    def wordToInf(self, text):
        process_pipeline = Pipeline(self.modelForInfinitive, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        wordInfo = process_pipeline.process(text).split('\n')[4].split('\t')
        if (wordInfo[3] == 'NUM'):
            return ('_NUM_' + ('x' * len(wordInfo[2])))
        else:
            return wordInfo[2]
    
    def text2Vec (self, text):
        self.vectorOfStr = torch.zeros(2000)
        self.nomberOfKnownWords = 0
        words = re.findall(r'[0-9A-Za-zА-Яа-я-.]+', text)
        for word in words:
            try :
                self.tryToUpgradeSelfVec(word)
            except Exception:
                for singleWord in re.findall(r"[\w']+", word):
                    try: 
                        self.tryToUpgradeSelfVec(singleWord)
                    except Exception:
                        if (self.needToSearch == True):
                            url = 'https://rusvectores.org/araneum_none_fasttextcbow_300_5_2018/' + singleWord + '/api/json/'
                            #url = 'https://rusvectores.org/tayga_none_fasttextcbow_300_10_2019/' + singleWord + '/api/json/'
                            response = requests.get(url)
                            for i in range (10):
                                try:
                                    sinonim = str(response.json())
                                    sinonim = sinonim.split("'")[5 + i * 2].split('_')[0]
                                    relationID = int(self.clasterOfWord[sinonim.lower()])
                                    self.clasterOfWord.update({singleWord: relationID})
                                    self.vectorOfStr[relationID] = self.vectorOfStr[relationID] + 1
                                    self.nomberOfKnownWords = self.nomberOfKnownWords + 1
                                    break
                                except Exception:
                                    continue
                                                 
        self.vectorOfStr = self.vectorOfStr / self.nomberOfKnownWords  
        return self.vectorOfStr
    
    def updateDLNetwork (self, text, themes, cicles=5, learing_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learing_rate)
        
        newLabel = torch.zeros (len(themes), dtype=torch.long)
        for i in range (len(themes)):
            newLabel[i] = self.idOfThemes[themes[i]]
        
        newData = torch.zeros(len(themes), 2000)
        vecOfText = self.text2Vec(text)
        for i in range (2000):
            for j in range (len(themes)):
                newData[j][i] = vecOfText[i]

        for i in range (cicles):    
            labels = newLabel
            optimizer.zero_grad()                             
            outputs = self.net(newData)                             
            loss = criterion(outputs, newLabel)                 
            loss.backward()                                   
            optimizer.step()  
            
        torch.save(self.net.state_dict(), self.sourceForDL)
        
    def themesOfTheText (self, text, themes):
        vecOfText = self.text2Vec(text)
        vecOfRelation = self.net(vecOfText)
        returnData = {}
        for i in range ( len(themes) ):
            index = self.idOfThemes[themes[i]]
            returnData.update({themes[i]: int ( 100 * float(vecOfRelation[index]))})
        return returnData
#--------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

textClassifier = DLTextClassifier('udpipe_syntagrus.model', 'clastersOfWords.txt', 'neurotext_with_vec.pkl')

allThemes = ['авто/мото', 'активный отдых','бизнес','домашние животные','здоровье','знакомство и общение','игры',
             'ИТ (компьютеры и софт)','кино','красота и мода','кулинария','культура и искусство','литература',
             'мобильная связь и интернет','музыка','наука и техника','недвижимость','новости и СМИ','безопасность',
             'образование','обустройство и ремонт','политика','продукты питания','промышленность','путешествия',
             'работа','развлечения','религия','дом и семья','спорт','страхование','телевидение','товары и услуги',
             'увлечения и хобби','финансы','фото','эзотерика','электроника и бытовая техника','эротика','юмор',
             'общество, гуманитарные науки','дизайн и графика']
#--------------------------------------------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/text_classifier/api', methods=['POST', 'PUT'])
def create_task():
    if (request.method == 'POST'):
        text = request.json['text']
        try:
            themes = request.json['themes']
        except Exception:
            themes = allThemes
        return jsonify(textClassifier.themesOfTheText(text, themes)), 201
    if (request.method == 'PUT'):
        text = request.json['text']
        themes = request.json['themes']
        textClassifier.updateDLNetwork(text, themes)
        return jsonify(textClassifier.themesOfTheText(text, allThemes)), 201

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
